"""TRUE_CALCULATOR의 서로게이트.

fit()이 database/를 통째로 메모리에 올리는 스냅샷 방식. 2층 구조:
  층1 (관측된 x)   반복 관측을 원소별 다수결로 합쳐 합의 배열을 반환
  층2 (미관측 x)   ordinal 공간의 정규화 L1 거리로 k최근접을 찾아 거리 가중 평균

옵티마이저는 result.score(= p_arrays의 합)를 연속 평가값으로 쓰고,
needs_test=True인 후보만 TRUE_CALCULATOR로 실제 평가한다.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import orjson

from util.ingest import ARRAY_KEYS, LIST_KEYS, SCALAR_KEYS, ARRAYS_DIRNAME, CATALOG_NAME

# optimizer.py는 이 저장소 밖에 있다(GA/SA 본체). import만 한다.
# optimizer.py가 import 경로에 없으면 이 모듈은 실행되지 않는다. util/ingest.py는
# optimizer에 의존하지 않으므로 단독 실행 가능.
from optimizer import code_to_ord, ord_to_code  # noqa: F401  (ord_to_code는 옵티마이저 측에서 사용)


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
    scalars: dict[str, np.ndarray]     # scalar key -> (n_obs,)
    lists: dict[str, object]           # list key -> 첫 관측


@dataclass
class MockResult:
    x: str
    layer: int
    arrays: dict[str, np.ndarray]      # bool 예측 (p >= 0.5)
    p_arrays: dict[str, np.ndarray]    # 원소별 [0, 1]
    lists: dict[str, object]
    scalars: dict[str, float]
    n_obs: int
    nearest_dist: float
    needs_test: bool
    score: float                       # p_arrays의 합 — 옵티마이저 평가값


@dataclass
class ValidationReport:
    ok: bool
    arrays: dict[str, dict] = field(default_factory=dict)
    scalars: dict[str, dict] = field(default_factory=dict)


# --------------------------------------------------------------------------
# 형태학 연산 (검증 계약의 erode/dilate)
# --------------------------------------------------------------------------
def _shift(mask: np.ndarray, axis: int, step: int) -> np.ndarray:
    """경계를 False로 채우는 시프트. np.roll은 wrap되므로 쓰지 않는다."""
    out = np.zeros_like(mask)
    src = [slice(None)] * mask.ndim
    dst = [slice(None)] * mask.ndim
    if step > 0:
        dst[axis], src[axis] = slice(step, None), slice(None, -step)
    else:
        dst[axis], src[axis] = slice(None, step), slice(-step, None)
    out[tuple(dst)] = mask[tuple(src)]
    return out


def dilate(mask: np.ndarray, d: int) -> np.ndarray:
    out = mask
    for _ in range(d):
        acc = out.copy()
        for axis in range(out.ndim):
            acc |= _shift(out, axis, 1)
            acc |= _shift(out, axis, -1)
        out = acc
    return out


def erode(mask: np.ndarray, d: int) -> np.ndarray:
    return ~dilate(~mask, d)


# --------------------------------------------------------------------------
# 검증 계약
# --------------------------------------------------------------------------
def array_bounds(stack: np.ndarray, cfg: MockConfig) -> tuple[np.ndarray, np.ndarray, str]:
    """예측이 들어가야 하는 (하한, 상한) 마스크."""
    if stack.shape[0] >= cfg.min_obs:
        return stack.all(axis=0), stack.any(axis=0), "intersection/union"
    consensus = stack.mean(axis=0) >= 0.5
    return erode(consensus, cfg.dilate_d), dilate(consensus, cfg.dilate_d), f"erode/dilate(d={cfg.dilate_d})"


def scalar_bounds(values: np.ndarray, cfg: MockConfig) -> tuple[float, float, str]:
    mean = float(values.mean())
    if values.shape[0] >= cfg.min_obs:
        # 관측이 1개면 표본표준편차가 정의되지 않는다(min_obs=1 설정에서 발생).
        sigma = float(values.std(ddof=1)) if values.shape[0] > 1 else 0.0
        return mean - cfg.z * sigma, mean + cfg.z * sigma, f"mean±{cfg.z}σ"
    return mean - cfg.abs_tol, mean + cfg.abs_tol, f"mean±{cfg.abs_tol}"


# --------------------------------------------------------------------------
class MOCKCalculator:
    def __init__(self, db_dir: str | Path = "database", config: MockConfig | None = None):
        self.db_dir = Path(db_dir)
        self.cfg = config or MockConfig()
        self.observed: dict[str, Observed] = {}
        self._x_list: list[str] = []
        self._ords: np.ndarray | None = None      # (n_x, n_vars)
        self._ranges: np.ndarray | None = None    # (n_vars,)

    # ---- fit -------------------------------------------------------------
    def fit(self) -> "MOCKCalculator":
        """database/를 메모리로 올린다. ingest 재실행 후 다시 호출한다."""
        rows = self._read_catalog()
        if not rows:
            raise RuntimeError(f"catalog가 비어 있다: {self.db_dir / CATALOG_NAME}")

        by_array: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_array[row["array_id"]].append(row)

        slices: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
        meta: dict[str, list[dict]] = defaultdict(list)
        for array_id, group in by_array.items():
            path = self.db_dir / ARRAYS_DIRNAME / f"{array_id}.npz"
            with np.load(path) as npz:
                for row in group:
                    pos = row["batch_pos"]
                    for key in ARRAY_KEYS:
                        slices[row["x"]][key].append(npz[key][pos].astype(bool))
                    meta[row["x"]].append(row)

        self.observed = {}
        for x, per_key in slices.items():
            stacks = {key: np.stack(per_key[key]) for key in ARRAY_KEYS}
            p = {key: stacks[key].mean(axis=0) for key in ARRAY_KEYS}
            consensus = {key: p[key] >= 0.5 for key in ARRAY_KEYS}
            scalars = {
                key: np.asarray([r[key] for r in meta[x]], dtype=float) for key in SCALAR_KEYS
            }
            self.observed[x] = Observed(
                x=x,
                ord_vec=np.asarray(code_to_ord(x), dtype=float),
                n_obs=stacks[ARRAY_KEYS[0]].shape[0],
                stacks=stacks,
                consensus=consensus,
                p=p,
                scalars=scalars,
                lists={key: meta[x][0][key] for key in LIST_KEYS},
            )

        self._x_list = sorted(self.observed)
        self._ords = np.stack([self.observed[x].ord_vec for x in self._x_list])
        spread = self._ords.max(axis=0) - self._ords.min(axis=0)
        self._ranges = np.where(spread > 0, spread, 1.0)  # 상수 변수는 거리에 기여하지 않음
        return self

    def _read_catalog(self) -> list[dict]:
        path = self.db_dir / CATALOG_NAME
        rows = []
        with path.open("rb") as f:
            for line in f:
                if line.strip():
                    rows.append(orjson.loads(line))
        return rows

    # ---- 거리 -------------------------------------------------------------
    def _distances(self, ord_vec: np.ndarray) -> np.ndarray:
        """정규화 L1: sum(|a-b| / range) / n_vars."""
        return (np.abs(self._ords - ord_vec) / self._ranges).sum(axis=1) / self._ords.shape[1]

    # ---- 예측 -------------------------------------------------------------
    def predict(self, x: str) -> MockResult:
        if self._ords is None:
            raise RuntimeError("fit()을 먼저 호출할 것")
        obs = self.observed.get(x)
        if obs is not None:
            return self._result(
                x, layer=1, p={k: obs.p[k] for k in ARRAY_KEYS},
                lists=obs.lists,
                scalars={k: float(obs.scalars[k].mean()) for k in SCALAR_KEYS},
                n_obs=obs.n_obs, nearest_dist=0.0,
            )
        return self._predict_layer2(x, np.asarray(code_to_ord(x), dtype=float))

    def _predict_layer2(self, x: str, ord_vec: np.ndarray, exclude: int | None = None) -> MockResult:
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
        scalars = {
            key: float(sum(wi * nb.scalars[key].mean() for wi, nb in zip(w, neighbors)))
            for key in SCALAR_KEYS
        }
        return self._result(
            x, layer=2, p=p,
            lists=neighbors[0].lists,  # 보간하지 않고 최근접 것을 그대로 쓴다
            scalars=scalars, n_obs=0, nearest_dist=float(dists[order[0]]),
        )

    def _result(self, x, layer, p, lists, scalars, n_obs, nearest_dist) -> MockResult:
        return MockResult(
            x=x, layer=layer,
            arrays={key: p[key] >= 0.5 for key in ARRAY_KEYS},
            p_arrays=p, lists=dict(lists), scalars=scalars,
            n_obs=n_obs, nearest_dist=nearest_dist,
            needs_test=nearest_dist > self.cfg.trust_dist,
            score=float(sum(p[key].sum() for key in ARRAY_KEYS)),
        )

    def evaluate(self, x: str) -> float:
        """옵티마이저용 연속 평가값."""
        return self.predict(x).score

    # ---- 검증 -------------------------------------------------------------
    def validate(self, x: str, result: MockResult) -> ValidationReport:
        """예측이 x의 실제 관측과 모순되지 않는지 본다. 관측된 x에만 쓸 수 있다."""
        obs = self.observed.get(x)
        if obs is None:
            raise KeyError(f"관측이 없는 x는 검증할 수 없다: {x}")

        report = ValidationReport(ok=True)
        for key in ARRAY_KEYS:
            lo, hi, rule = array_bounds(obs.stacks[key], self.cfg)
            pred = result.arrays[key]
            below = int((lo & ~pred).sum())
            above = int((pred & ~hi).sum())
            ok = below == 0 and above == 0
            report.arrays[key] = {
                "ok": ok, "below": below, "above": above,
                "n_obs": obs.n_obs, "rule": rule,
            }
            report.ok &= ok
        for key in SCALAR_KEYS:
            lo, hi, rule = scalar_bounds(obs.scalars[key], self.cfg)
            value = result.scalars[key]
            ok = lo <= value <= hi
            report.scalars[key] = {
                "ok": ok, "value": value, "lo": lo, "hi": hi,
                "n_obs": obs.n_obs, "rule": rule,
            }
            report.ok &= ok
        return report

    def self_check(self) -> dict:
        """합의가 자기 관측 범위를 통과하는지. 구조적으로 100%여야 한다."""
        failures = []
        for x in self._x_list:
            if not self.validate(x, self.predict(x)).ok:
                failures.append(x)
        return {
            "n_x": len(self._x_list),
            "n_fail": len(failures),
            "pass_rate": 1.0 - len(failures) / len(self._x_list),
            "failures": failures[:10],
        }

    def loo_check(self, limit: int | None = None) -> dict:
        """leave-one-out으로 층2의 예측력을 잰다."""
        if len(self._x_list) < 2:
            return {"n_x": len(self._x_list), "note": "관측 x가 2개 미만이라 생략"}

        targets = self._x_list if limit is None else self._x_list[:limit]
        n_pass, n_needs_test, agree = 0, 0, []
        for x in targets:
            i = self._x_list.index(x)
            pred = self._predict_layer2(x, self.observed[x].ord_vec, exclude=i)
            if self.validate(x, pred).ok:
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="MOCKCalculator fit + 자가 점검")
    ap.add_argument("--db-dir", default="database")
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
