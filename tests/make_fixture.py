"""스펙만으로 소형 합성 raw/*.jsonl을 만든다 (실제 데이터 없이 테스트하기 위한 것).

실제 실험의 키 이름/차원/수치는 모르므로 util/_update_database.py 헤더의
플레이스홀더 상수를 그대로 따르고, 내용은 스펙에 적힌 성질(볼록 blob, 경계
noise)만 흉내낸다.

포함하는 케이스:
  - 두 줄 분할 기록 (배열 줄 + 스칼라 줄)
  - 파일 간 iteration 번호 리셋
  - 같은 x의 반복 관측 (blob 경계가 1~2원소 흔들림)
  - 축 0/1이 뒤바뀐 run 하나 (run_b)
  - 미완성 iteration 하나 (run_c:3 — 배열 줄만 있고 스칼라 줄이 없음)
  - 쓰다 만 마지막 줄 하나 (run_d — 실시간 append 중인 파일)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import orjson

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util._update_database import (  # noqa: E402
    ARRAY_KEYS,
    AXIS1_SIZE,
    LIST_KEYS,
    SCALAR_KEYS,
    X_KEY,
)

SHAPE = (3, AXIS1_SIZE, 2, 2, 2)   # 내부 5D. 축1이 고정 크기 AXIS1_SIZE.
LIST_LEN = 3
_GRID = np.indices(SHAPE).astype(float)

RUN_A_CODES = ["AAA", "ABC", "BCA", "CAB", "BBD", "DCB"]
RUN_C_CODES = ["EAB", "ACE", "BDA", "CCC"]
RUN_D_CODES = ["DDE", "EEA"]


def code_to_ord_stub(code: str) -> list[int]:
    """테스트용 stub. 실제 optimizer.code_to_ord 자리에 주입한다."""
    return [ord(c) for c in code]


def _seed(code: str) -> int:
    # hash()는 실행마다 달라지므로 쓰지 않는다.
    return int.from_bytes(code.encode(), "big") % (2**31)


def _blob(code: str) -> tuple[np.ndarray, np.ndarray, float]:
    """코드마다 결정적인 볼록 blob (구 형태)."""
    rng = np.random.default_rng(_seed(code))
    center = np.array(SHAPE, dtype=float) / 2 - 0.5 + rng.uniform(-0.4, 0.4, size=len(SHAPE))
    radius = 1.3 + rng.uniform(0.0, 0.6)
    dist = np.sqrt(sum((_GRID[i] - center[i]) ** 2 for i in range(len(SHAPE))))
    return dist <= radius, dist, radius


def _observe(code: str, rng: np.random.Generator) -> np.ndarray:
    """한 번의 관측. 경계 원소 1~2개가 흔들린다(noise)."""
    base, dist, radius = _blob(code)
    boundary = np.flatnonzero(np.abs(dist.ravel() - radius) <= 0.7)
    obs = base.ravel().copy()
    if boundary.size:
        n_flip = min(int(rng.integers(1, 3)), boundary.size)
        obs[rng.choice(boundary, size=n_flip, replace=False)] ^= True
    return obs.reshape(SHAPE)


def _list_y(code: str, rng: np.random.Generator, offset: float) -> list[float]:
    """평균이 최소화 대상인 list y. 코드에 따라 값이 달라져야 층2가 의미를 갖는다."""
    base = offset + sum(code_to_ord_stub(code)) / 100.0
    return [round(float(v), 6) for v in rng.normal(base, 0.02, size=LIST_LEN)]


def _iteration_lines(codes, rng, swapped: bool) -> list[dict]:
    """한 iteration을 배열 줄 + 스칼라 줄로 쪼갠다."""
    array_line = {X_KEY: list(codes)}
    for ki, key in enumerate(ARRAY_KEYS):
        # 배열 2개는 서로 다른 blob이어야 한다. 코드에 접미사를 붙여 다른 씨앗을 쓴다.
        stack = np.stack([_observe(c + "#" * ki, rng) for c in codes])
        if swapped:
            stack = np.swapaxes(stack, 1, 2)  # 내부 축 0/1 뒤바뀜
        array_line[key] = stack.tolist()
    for ki, key in enumerate(LIST_KEYS):
        array_line[key] = [_list_y(c, rng, offset=1.0 + ki) for c in codes]
    for key in SCALAR_KEYS:
        array_line[key] = None

    scalar_line = {X_KEY: None}
    for key in ARRAY_KEYS + LIST_KEYS:
        scalar_line[key] = None
    for ki, key in enumerate(SCALAR_KEYS):
        scalar_line[key] = round(float(10.0 + ki + rng.normal(0, 0.05)), 6)
    return [array_line, scalar_line]


def _write(path: Path, lines: list[tuple[int, dict]], torn: bytes | None = None) -> None:
    with path.open("wb") as f:
        for iteration, payload in lines:
            f.write(orjson.dumps({str(iteration): payload}))
            f.write(b"\n")
        if torn is not None:
            f.write(torn)  # 개행 없이 잘린 줄 — 실험이 아직 쓰는 중


def make_fixture(raw_dir: str | Path, batch: int = 2) -> dict:
    """raw_dir에 합성 run 파일들을 쓰고, 테스트가 기대값으로 쓸 요약을 돌려준다."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    def run(codes, seed, swapped=False, n_iter=None):
        rng = np.random.default_rng(seed)
        n_iter = n_iter if n_iter is not None else len(codes) // batch
        out = []
        for it in range(1, n_iter + 1):
            chunk = codes[(it - 1) * batch: it * batch]
            for payload in _iteration_lines(chunk, rng, swapped):
                out.append((it, payload))  # iteration 번호는 run마다 1부터
        return out

    _write(raw_dir / "run_a.jsonl", run(RUN_A_CODES, seed=1))
    # 같은 코드를 다시 관측한 run. 축 0/1이 뒤바뀐 채 기록되었다.
    _write(raw_dir / "run_b.jsonl", run(RUN_A_CODES, seed=2, swapped=True))

    # run_c: 완성된 iteration 2개 + 미완성 iteration 1개(배열 줄만).
    lines_c = run(RUN_C_CODES, seed=3)
    rng_c = np.random.default_rng(30)
    lines_c.append((3, _iteration_lines(["ABA", "BAB"], rng_c, swapped=False)[0]))
    _write(raw_dir / "run_c.jsonl", lines_c)

    # run_d: 완성된 iteration 1개 + 쓰다 만 마지막 줄.
    lines_d = run(RUN_D_CODES, seed=4)
    rng_d = np.random.default_rng(40)
    torn_payload = orjson.dumps({"2": _iteration_lines(["CDC", "DAD"], rng_d, False)[0]})
    _write(raw_dir / "run_d.jsonl", lines_d, torn=torn_payload[: len(torn_payload) // 3])

    return {
        "raw_dir": raw_dir,
        "shape": SHAPE,
        "batch": batch,
        "complete_array_ids": [
            "run_a:1", "run_a:2", "run_a:3",
            "run_b:1", "run_b:2", "run_b:3",
            "run_c:1", "run_c:2",
            "run_d:1",
        ],
        "incomplete_array_ids": ["run_c:3"],
        "torn_file": "run_d.jsonl",
        "repeated_codes": list(RUN_A_CODES),          # run_a + run_b -> 관측 2회
        "single_codes": RUN_C_CODES + RUN_D_CODES,    # 관측 1회
        "swapped_run": "run_b",
    }


def complete_torn_line(raw_dir: str | Path) -> list[str]:
    """쓰다 만 마지막 줄을 실험이 마저 쓴 상황을 만든다 (run_d:2가 완성된다).

    잘린 줄을 버리고, 같은 씨앗으로 같은 iteration을 배열 줄 + 스칼라 줄로 다시
    쓴다. 완성된 코드 목록을 돌려준다.
    """
    path = Path(raw_dir) / "run_d.jsonl"
    data = path.read_bytes()
    head = data[: data.rfind(b"\n") + 1]     # 개행이 없는 잘린 줄만 떨어져 나간다

    codes = ["CDC", "DAD"]
    rng = np.random.default_rng(40)          # make_fixture가 쓴 것과 같은 씨앗
    with path.open("wb") as f:
        f.write(head)
        for payload in _iteration_lines(codes, rng, swapped=False):
            f.write(orjson.dumps({"2": payload}))
            f.write(b"\n")
    return codes


def complete_incomplete_iteration(raw_dir: str | Path, seed: int = 99) -> None:
    """미완성이던 run_c:3에 스칼라 줄이 뒤늦게 도착한 상황을 만든다."""
    rng = np.random.default_rng(seed)
    scalar_line = _iteration_lines(["ABA", "BAB"], rng, swapped=False)[1]
    with (Path(raw_dir) / "run_c.jsonl").open("ab") as f:
        f.write(orjson.dumps({"3": scalar_line}))
        f.write(b"\n")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "raw"
    info = make_fixture(target)
    print(f"생성 완료: {sorted(p.name for p in Path(target).glob('*.jsonl'))}")
    print(f"완성 iteration {len(info['complete_array_ids'])}개, "
          f"미완성 {len(info['incomplete_array_ids'])}개, 쓰다 만 줄 1개")
