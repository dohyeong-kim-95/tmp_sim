"""TRUE_CALCULATOR의 서로게이트. fit()이 database/를 메모리에 올린다.

  층1 (관측된 x)   반복 관측의 원소별 다수결
  층2 (미관측 x)   ordinal 정규화 L1 거리로 k최근접 가중 평균

다목적이라 하나의 점수로 합치지 않는다 — 가중 결합은 옵티마이저 책임.
needs_test=True면 TRUE_CALCULATOR로 실제 평가한다.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import orjson

from util._update_database import (
    ARRAY_KEYS,
    ARRAYS_DIRNAME,
    CATALOG_NAME,
    DB_DIR,
    LIST_KEYS,
    SCALAR_KEYS,
)

ARRAY_SUM_KEY = "array_sum"


def objective_key(list_key):
    """list y 키에서 목적함수 이름을 만든다."""
    return f"{list_key}_mean"


@dataclass(frozen=True)
class MockConfig:
    k: int = 5              # 층2 최근접 이웃 수
    trust_dist: float = 0.05  # 최근접 거리가 이보다 크면 needs_test
    min_obs: int = 3        # 검증 계약에서 "관측 충분"의 기준
    dilate_d: int = 1       # 관측 부족 시 erode/dilate 거리 (Manhattan)
    z: float = 3.0          # 스칼라 허용 범위 mean ± z·σ
    abs_tol: float = 1.0    # 관측 부족 시 스칼라 허용 범위 mean ± abs_tol
    eps: float = 1e-9       # 거리 가중치 0 나눗셈 방지


@dataclass
class Observed:
    """한 x에 대한 모든 반복 관측과 그 합의."""
    x: str
    ord_vec: np.ndarray
    n_obs: int
    stacks: dict[str, np.ndarray]      # array key -> (n_obs, *5D) bool
    consensus: dict[str, np.ndarray]   # array key -> 5D bool (원소별 다수결)
    p: dict[str, np.ndarray]           # array key -> 5D float (True 비율)
    list_means: dict[str, np.ndarray]  # list key -> (n_obs,) 관측별 mean(list)
    scalars: dict[str, np.ndarray]     # scalar key -> (n_obs,)


@dataclass
class MockResult:
    x: str
    layer: int
    arrays: dict[str, np.ndarray]      # bool 예측 (p >= 0.5)
    p_arrays: dict[str, np.ndarray]    # 원소별 [0, 1]
    objectives: dict[str, float]       # array_sum(최대화) + <list key>_mean(최소화)
    scalars: dict[str, float]          # batch 공통 설정값 — 목적함수가 아니다
    n_obs: int
    nearest_dist: float
    needs_test: bool


# --------------------------------------------------------------------------
# 형태학 연산 (검증 계약의 erode/dilate)
# --------------------------------------------------------------------------
def _shift(mask, axis, step):
    """경계를 False로 채우는 시프트. np.roll은 wrap되어 못 쓴다."""
    out = np.zeros_like(mask)
    src = [slice(None)] * mask.ndim
    dst = [slice(None)] * mask.ndim
    if step > 0:
        dst[axis], src[axis] = slice(step, None), slice(None, -step)
    else:
        dst[axis], src[axis] = slice(None, step), slice(-step, None)
    out[tuple(dst)] = mask[tuple(src)]
    return out


def dilate(mask, d):
    out = mask
    for _ in range(d):
        acc = out.copy()
        for axis in range(out.ndim):
            acc |= _shift(out, axis, 1)
            acc |= _shift(out, axis, -1)
        out = acc
    return out


def erode(mask, d):
    return ~dilate(~mask, d)


# --------------------------------------------------------------------------
# 검증 계약
# --------------------------------------------------------------------------
def array_bounds(stack, cfg):
    """관측 스택에서 예측이 들어가야 할 (하한, 상한, 규칙명)을 돌려준다."""
    if stack.shape[0] >= cfg.min_obs:
        return stack.all(axis=0), stack.any(axis=0), "intersection/union"
    consensus = stack.mean(axis=0) >= 0.5
    return erode(consensus, cfg.dilate_d), dilate(consensus, cfg.dilate_d), f"erode/dilate(d={cfg.dilate_d})"


def scalar_bounds(values, cfg):
    """관측값들에서 예측이 들어가야 할 (하한, 상한, 규칙명)을 돌려준다."""
    mean = float(values.mean())
    if values.shape[0] >= cfg.min_obs:
        # 관측 1개면 표본표준편차가 정의되지 않는다.
        sigma = float(values.std(ddof=1)) if values.shape[0] > 1 else 0.0
        return mean - cfg.z * sigma, mean + cfg.z * sigma, f"mean±{cfg.z}σ"
    return mean - cfg.abs_tol, mean + cfg.abs_tol, f"mean±{cfg.abs_tol}"


# --------------------------------------------------------------------------
class MOCKCalculator:
    def __init__(
        self,
        db_dir=None,
        config=None,
        code_to_ord=None,
    ):
        """code_to_ord 미지정 시 optimizer에서 늦게 import한다 (테스트는 stub 주입)."""
        self.db_dir = Path(db_dir) if db_dir is not None else DB_DIR
        self.cfg = config or MockConfig()
        self._code_to_ord = code_to_ord
        self.observed = {}          # x -> Observed
        self._x_list = []           # 정렬된 x 목록. _ords의 행 순서
        self._ords = None           # (n_x, n_vars). fit()이 채운다
        self._ranges = None         # (n_vars,) 거리 정규화용

    def _ord_vec(self, x):
        if self._code_to_ord is None:
            from optimizer import code_to_ord  # optimizer.py는 이 저장소 밖에 있다

            self._code_to_ord = code_to_ord
        return np.asarray(self._code_to_ord(x), dtype=float)

    # ---- 적재 -------------------------------------------------------------
    def _load_database(self, db_dir=None):
        """db_dir을 읽어 x -> {"slices": array key별 관측 리스트, "rows": catalog 줄}을 돌려준다."""
        db_dir = Path(db_dir) if db_dir is not None else self.db_dir

        catalog_path = db_dir / CATALOG_NAME
        array_id_to_catalog_rows = defaultdict(list)   # array_id -> catalog 줄들
        with catalog_path.open("rb") as f:
            for line in f:
                if line.strip():
                    row = orjson.loads(line)
                    array_id_to_catalog_rows[row["array_id"]].append(row)
        if not array_id_to_catalog_rows:
            raise RuntimeError(f"catalog가 비어 있다: {catalog_path}")

        loaded = defaultdict(lambda: {"slices": defaultdict(list), "rows": []})
        for array_id, group in array_id_to_catalog_rows.items():
            with np.load(db_dir / ARRAYS_DIRNAME / f"{array_id}.npz") as npz:
                batched = {key: npz[key] for key in ARRAY_KEYS}   # npz[key]는 캐시 안 되므로 루프 밖에서
                for row in group:
                    per_x = loaded[row["x"]]
                    for key in ARRAY_KEYS:
                        per_x["slices"][key].append(batched[key][row["batch_pos"]].astype(bool))
                    per_x["rows"].append(row)
        return dict(loaded)

    # ---- fit -------------------------------------------------------------
    def fit(self):
        """database/를 x별 Observed와 거리 계산용 ordinal 행렬로 올리고 self를 돌려준다."""
        self.observed = {}
        for x, raw in self._load_database().items():
            stacks = {key: np.stack(raw["slices"][key]) for key in ARRAY_KEYS}
            p = {key: stacks[key].mean(axis=0) for key in ARRAY_KEYS}
            self.observed[x] = Observed(
                x=x,
                ord_vec=self._ord_vec(x),
                n_obs=stacks[ARRAY_KEYS[0]].shape[0],
                stacks=stacks,
                consensus={key: p[key] >= 0.5 for key in ARRAY_KEYS},
                p=p,
                # list y는 평균만 쓰므로 원소별 합의를 하지 않는다.
                list_means={
                    key: np.asarray([float(np.mean(r[key])) for r in raw["rows"]], dtype=float)
                    for key in LIST_KEYS
                },
                scalars={
                    key: np.asarray([r[key] for r in raw["rows"]], dtype=float)
                    for key in SCALAR_KEYS
                },
            )
        self._x_list = sorted(self.observed)
        self._ords = np.stack([self.observed[x].ord_vec for x in self._x_list])
        spread = self._ords.max(axis=0) - self._ords.min(axis=0)
        self._ranges = np.where(spread > 0, spread, 1.0)  # 상수 변수는 거리 기여 0
        return self

    # ---- 거리 -------------------------------------------------------------
    def _distances(self, ord_vec):
        """ord_vec과 모든 관측 x 사이의 정규화 L1 거리 배열을 돌려준다."""
        return (np.abs(self._ords - ord_vec) / self._ranges).sum(axis=1) / self._ords.shape[1]

    # ---- 예측 -------------------------------------------------------------
    def predict(self, x):
        if self._ords is None:
            raise RuntimeError("fit()을 먼저 호출할 것")
        obs = self.observed.get(x)
        if obs is not None:
            return self._result(
                x, layer=1,
                p={k: obs.p[k] for k in ARRAY_KEYS},
                list_means={k: float(obs.list_means[k].mean()) for k in LIST_KEYS},
                scalars={k: float(obs.scalars[k].mean()) for k in SCALAR_KEYS},
                n_obs=obs.n_obs, nearest_dist=0.0,
            )
        return self._predict_layer2(x, self._ord_vec(x))

    def _predict_layer2(self, x, ord_vec, exclude=None):
        dists = self._distances(ord_vec)
        if exclude is not None:
            dists = dists.copy()
            dists[exclude] = np.inf
        order = np.argsort(dists)[: self.cfg.k]
        order = order[np.isfinite(dists[order])]
        if order.size == 0:
            raise RuntimeError("이웃이 없다. 관측이 최소 1개는 있어야 한다.")

        w = 1.0 / (dists[order] + self.cfg.eps)
        w /= w.sum()
        neighbors = [self.observed[self._x_list[i]] for i in order]

        p = {
            key: sum(wi * nb.consensus[key] for wi, nb in zip(w, neighbors))
            for key in ARRAY_KEYS
        }
        list_means = {
            key: float(sum(wi * nb.list_means[key].mean() for wi, nb in zip(w, neighbors)))
            for key in LIST_KEYS
        }
        scalars = {
            key: float(sum(wi * nb.scalars[key].mean() for wi, nb in zip(w, neighbors)))
            for key in SCALAR_KEYS
        }
        return self._result(
            x, layer=2, p=p, list_means=list_means, scalars=scalars,
            n_obs=0, nearest_dist=float(dists[order[0]]),
        )

    def _result(self, x, layer, p, list_means, scalars, n_obs, nearest_dist):
        objectives = {ARRAY_SUM_KEY: float(sum(p[key].sum() for key in ARRAY_KEYS))}
        for key in LIST_KEYS:
            objectives[objective_key(key)] = list_means[key]
        return MockResult(
            x=x, layer=layer,
            arrays={key: p[key] >= 0.5 for key in ARRAY_KEYS},
            p_arrays=p, objectives=objectives, scalars=scalars,
            n_obs=n_obs, nearest_dist=nearest_dist,
            needs_test=nearest_dist > self.cfg.trust_dist,
        )

    def objectives(self, x):
        """x의 목적별 값 dict. array_sum은 최대화, <list key>_mean은 최소화."""
        return self.predict(x).objectives

    # ---- 검증 -------------------------------------------------------------
    def validate(self, x, result):
        """관측된 x의 예측을 관측과 대조해 {"ok", "arrays", "objectives", "scalars"}를 돌려준다."""
        obs = self.observed.get(x)
        if obs is None:
            raise KeyError(f"관측이 없는 x는 검증할 수 없다: {x}")

        report = {"ok": True, "arrays": {}, "objectives": {}, "scalars": {}}
        for key in ARRAY_KEYS:
            lo, hi, rule = array_bounds(obs.stacks[key], self.cfg)
            pred = result.arrays[key]
            below, above = int((lo & ~pred).sum()), int((pred & ~hi).sum())
            report["arrays"][key] = {
                "ok": below == 0 and above == 0, "below": below, "above": above,
                "n_obs": obs.n_obs, "rule": rule,
            }
        for key in LIST_KEYS:
            report["objectives"][objective_key(key)] = self._check_scalar(
                obs.list_means[key], result.objectives[objective_key(key)], obs.n_obs
            )
        for key in SCALAR_KEYS:
            report["scalars"][key] = self._check_scalar(
                obs.scalars[key], result.scalars[key], obs.n_obs
            )
        report["ok"] = all(
            entry["ok"]
            for section in ("arrays", "objectives", "scalars")
            for entry in report[section].values()
        )
        return report

    def _check_scalar(self, values, value, n_obs):
        lo, hi, rule = scalar_bounds(values, self.cfg)
        return {
            "ok": bool(lo <= value <= hi), "value": value,
            "lo": lo, "hi": hi, "n_obs": n_obs, "rule": rule,
        }

    # ---- 자가점검 ---------------------------------------------------------
    def self_check(self):
        """모든 관측 x의 층1 예측이 자기 관측 범위를 통과하는지 요약 dict로 돌려준다."""
        failures = [x for x in self._x_list if not self.validate(x, self.predict(x))["ok"]]
        return {
            "n_x": len(self._x_list),
            "n_fail": len(failures),
            "pass_rate": 1.0 - len(failures) / len(self._x_list),
            "failures": failures[:10],
        }

    def loo_check(self, limit=None):
        """각 x를 이웃 풀에서 빼고 층2로 예측해 그 예측력을 요약 dict로 돌려준다."""
        if len(self._x_list) < 2:
            return {"n_x": len(self._x_list), "note": "관측 x가 2개 미만이라 생략"}

        targets = self._x_list if limit is None else self._x_list[:limit]
        n_pass, n_needs_test, agree = 0, 0, []
        for i, x in enumerate(targets):
            pred = self._predict_layer2(x, self.observed[x].ord_vec, exclude=i)
            if self.validate(x, pred)["ok"]:
                n_pass += 1
            n_needs_test += int(pred.needs_test)
            for key in ARRAY_KEYS:
                truth = self.observed[x].consensus[key]
                agree.append(float((pred.arrays[key] == truth).mean()))
        n = len(targets)
        return {
            "n_x": n,
            "pass_rate": n_pass / n,
            "needs_test_rate": n_needs_test / n,
            "mean_elementwise_agreement": float(np.mean(agree)),
        }


def main(argv=None):
    ap = argparse.ArgumentParser(description="MOCKCalculator fit + 자가 점검")
    ap.add_argument("--db-dir", default=DB_DIR)
    ap.add_argument("--k", type=int, default=MockConfig.k)
    ap.add_argument("--trust-dist", type=float, default=MockConfig.trust_dist)
    ap.add_argument("--min-obs", type=int, default=MockConfig.min_obs)
    args = ap.parse_args(argv)

    mock = MOCKCalculator(
        args.db_dir,
        MockConfig(k=args.k, trust_dist=args.trust_dist, min_obs=args.min_obs),
    ).fit()

    print(f"[mock] 관측 x {len(mock.observed)}개")
    print(f"[mock] self_check {json.dumps(mock.self_check(), ensure_ascii=False)}")
    print(f"[mock] loo_check  {json.dumps(mock.loo_check(), ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
