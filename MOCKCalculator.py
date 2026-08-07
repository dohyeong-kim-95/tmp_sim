"""TRUE_CALCULATOR의 서로게이트. fit()이 database/를 메모리에 올린다.

  층1 (관측된 x)   반복 관측의 원소별 다수결
  층2 (미관측 x)   ordinal 정규화 L1 거리로 k최근접 가중 평균

다목적이라 하나의 점수로 합치지 않는다 — 가중 결합은 옵티마이저 책임.
needs_test=True면 TRUE_CALCULATOR로 실제 평가한다.
"""

from __future__ import annotations

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
    eps: float = 1e-9       # 거리 가중치 0 나눗셈 방지


@dataclass
class Observed:
    """한 x의 반복 관측 요약."""
    n_obs: int
    p: dict[str, np.ndarray]   # array key -> 5D float (관측 중 True 비율)
    means: dict[str, float]    # list key -> mean(list)의 평균, scalar key -> 그 평균


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


class MOCKCalculator:
    def __init__(self, db_dir=None, config=None, code_to_ord=None):
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

    # ---- 1. 뼈대: catalog를 x별로 묶어 배열 없는 Observed를 만든다 ----------
    def _read_catalog(self, db_dir):
        """catalog를 읽어 (x -> p가 빈 Observed, array_id -> 그 npz가 담당하는 줄들)을 돌려준다."""
        rows_by_x = defaultdict(list)
        rows_by_array_id = defaultdict(list)
        with (db_dir / CATALOG_NAME).open("rb") as f:
            for line in f:
                if line.strip():
                    row = orjson.loads(line)
                    rows_by_x[row["x"]].append(row)
                    rows_by_array_id[row["array_id"]].append(row)
        if not rows_by_x:
            raise RuntimeError(f"catalog가 비어 있다: {db_dir / CATALOG_NAME}")

        observed = {
            x: Observed(
                n_obs=len(rows),
                p={key: [] for key in ARRAY_KEYS},   # 2단계가 쌓고 3단계가 접는다
                means={
                    # list y는 평균만 쓰므로 원소별 합의를 하지 않는다.
                    **{key: float(np.mean([np.mean(r[key]) for r in rows])) for key in LIST_KEYS},
                    **{key: float(np.mean([r[key] for r in rows])) for key in SCALAR_KEYS},
                },
            )
            for x, rows in rows_by_x.items()
        }
        return observed, rows_by_array_id

    # ---- 2. 배열 주입: npz를 array_id당 한 번만 열어 슬라이스를 쌓는다 -------
    def _load_arrays(self, db_dir, observed, rows_by_array_id):
        """각 npz의 batch_pos 슬라이스를 해당 x의 Observed.p에 쌓는다 (반환 없음)."""
        for array_id, rows in rows_by_array_id.items():
            with np.load(db_dir / ARRAYS_DIRNAME / f"{array_id}.npz") as npz:
                batched = {key: npz[key] for key in ARRAY_KEYS}   # npz[key]는 캐시 안 되므로 루프 밖에서
                for row in rows:
                    p = observed[row["x"]].p
                    for key in ARRAY_KEYS:
                        p[key].append(batched[key][row["batch_pos"]].astype(bool))

    # ---- 3. 요약: 쌓인 슬라이스를 원소별 True 비율로 접는다 -----------------
    def _build_p(self, observed):
        """각 Observed.p의 슬라이스 리스트를 원소별 True 비율 5D 배열로 바꾼다 (반환 없음)."""
        for obs in observed.values():
            obs.p = {key: np.stack(obs.p[key]).mean(axis=0) for key in ARRAY_KEYS}

    # ---- fit -------------------------------------------------------------
    def fit(self):
        """database/를 x별 Observed와 거리 계산용 ordinal 행렬로 올리고 self를 돌려준다."""
        observed, rows_by_array_id = self._read_catalog(self.db_dir)
        self._load_arrays(self.db_dir, observed, rows_by_array_id)
        self._build_p(observed)

        self.observed = observed
        self._x_list = sorted(self.observed)
        self._ords = np.stack([self._ord_vec(x) for x in self._x_list])
        spread = self._ords.max(axis=0) - self._ords.min(axis=0)
        self._ranges = np.where(spread > 0, spread, 1.0)  # 상수 변수는 거리 기여 0
        return self

    # ---- 거리 -------------------------------------------------------------
    def _distances(self, ord_vec):
        """ord_vec과 모든 관측 x 사이의 정규화 L1 거리 배열을 돌려준다."""
        return (np.abs(self._ords - ord_vec) / self._ranges).sum(axis=1) / self._ords.shape[1]

    # ---- 예측 -------------------------------------------------------------
    def predict(self, x):
        """x의 MockResult를 돌려준다. 관측된 x면 층1, 미관측이면 층2."""
        if self._ords is None:
            raise RuntimeError("fit()을 먼저 호출할 것")
        obs = self.observed.get(x)
        if obs is not None:
            return self._result(x, 1, obs.p, obs.means, obs.n_obs, 0.0)
        return self._predict_layer2(x, self._ord_vec(x))

    def _predict_layer2(self, x, ord_vec, exclude=None):
        """k최근접 관측의 거리 가중 평균으로 미관측 x의 MockResult를 만든다."""
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
            key: sum(wi * (nb.p[key] >= 0.5) for wi, nb in zip(w, neighbors))
            for key in ARRAY_KEYS
        }
        means = {
            key: float(sum(wi * nb.means[key] for wi, nb in zip(w, neighbors)))
            for key in LIST_KEYS + SCALAR_KEYS
        }
        return self._result(x, 2, p, means, 0, float(dists[order[0]]))

    def _result(self, x, layer, p, means, n_obs, nearest_dist):
        """층이 만든 p와 means를 MockResult로 조립한다."""
        objectives = {ARRAY_SUM_KEY: float(sum(p[key].sum() for key in ARRAY_KEYS))}
        for key in LIST_KEYS:
            objectives[objective_key(key)] = means[key]
        return MockResult(
            x=x, layer=layer,
            arrays={key: p[key] >= 0.5 for key in ARRAY_KEYS},
            p_arrays=p, objectives=objectives,
            scalars={key: means[key] for key in SCALAR_KEYS},
            n_obs=n_obs, nearest_dist=nearest_dist,
            needs_test=nearest_dist > self.cfg.trust_dist,
        )
