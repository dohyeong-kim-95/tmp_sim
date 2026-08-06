"""raw/*.jsonl -> database/ ingest 파이프라인.

한 줄 = {"<iteration>": {"<키>": <값>}}. 한 iteration의 정보가 두 줄(배열 줄 +
스칼라 줄)에 나뉘어 기록되므로 null이 아닌 값으로 병합한다.

산출물:
  database/catalog.jsonl          x 하나당 한 줄
  database/arrays/{array_id}.npz  5D bool 배열을 batch째로 저장

멱등: catalog에 이미 있는 array_id는 스킵. 필수 키가 안 모인 iteration도 스킵.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import orjson

# --------------------------------------------------------------------------
# 플레이스홀더 — 실제 로그의 키 이름/차원으로 교체할 것.
# --------------------------------------------------------------------------
X_KEY = "x"
ARRAY_KEYS = ("yARR1_key", "yARR2_key")      # bool 5D array (batch 포함 6D로 기록됨)
LIST_KEYS = ("yLST1_key", "yLST2_key")       # x 하나당 list
SCALAR_KEYS = ("y1_cfg_key", "y2_cfg_key")     # batch 공통 설정값

# 배열 내부 축1의 고정 크기. 축 0/1 뒤바뀜 판정에 쓰인다.
AXIS1_SIZE = 4

REQUIRED_KEYS = (X_KEY,) + ARRAY_KEYS + LIST_KEYS + SCALAR_KEYS

CATALOG_NAME = "catalog.jsonl"
ARRAYS_DIRNAME = "arrays"


class IngestError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# 읽기
# --------------------------------------------------------------------------
def load_done_ids(catalog_path: Path) -> set[str]:
    """catalog에 이미 기록된 array_id 집합. 멱등성의 근거."""
    done: set[str] = set()
    if not catalog_path.exists():
        return done
    with catalog_path.open("rb") as f:
        for line in f:
            if line.strip():
                done.add(orjson.loads(line)["array_id"])
    return done


def read_run(path: Path) -> dict[int, dict]:
    """한 run 파일을 iteration -> 병합된 레코드로 읽는다.

    실험이 실시간으로 append하는 파일이라 마지막 줄은 쓰다 만 상태일 수 있다.
    그건 정상 상황이므로 경고만 하고 넘긴다(다음 실행에서 완성된다).
    마지막이 아닌 줄이 깨졌다면 진짜 손상이므로 중단한다.
    """
    records: dict[int, dict] = {}
    with path.open("rb") as f:
        lines = f.readlines()
    last_lineno = max((i for i, l in enumerate(lines, 1) if l.strip()), default=0)

    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            obj = orjson.loads(line)
        except orjson.JSONDecodeError as e:
            if lineno == last_lineno:
                print(
                    f"[ingest] 경고: {path}:{lineno} 마지막 줄이 쓰다 만 상태 — 스킵 ({e})",
                    file=sys.stderr,
                )
                continue
            raise IngestError(f"{path}:{lineno} JSON 파싱 실패: {e}") from e
        for iteration, fields in obj.items():
            rec = records.setdefault(int(iteration), {})
            for key, value in fields.items():
                if value is not None:
                    rec[key] = value
    return records


# --------------------------------------------------------------------------
# 정규화
# --------------------------------------------------------------------------
def orient(arr: np.ndarray, array_id: str, key: str) -> np.ndarray:
    """내부 축 0/1이 뒤바뀐 run을 통일한다.

    기록 배열은 (batch, i0, i1, i2, i3, i4). 정상이면 i1 == AXIS1_SIZE.
    """
    if arr.ndim != 6:
        raise IngestError(
            f"{array_id}/{key}: (batch + 5D) = 6D를 기대했으나 ndim={arr.ndim}, "
            f"shape={arr.shape}"
        )
    i0, i1 = arr.shape[1], arr.shape[2]
    if i1 == AXIS1_SIZE and i0 != AXIS1_SIZE:
        return arr
    if i0 == AXIS1_SIZE and i1 != AXIS1_SIZE:
        return np.swapaxes(arr, 1, 2)
    raise IngestError(
        f"{array_id}/{key}: 축 0/1 뒤바뀜 판정 불가. shape={arr.shape}, "
        f"AXIS1_SIZE={AXIS1_SIZE}. 조용히 통과시키면 database가 오염되므로 중단한다."
    )


def build_arrays(rec: dict, array_id: str, batch: int) -> dict[str, np.ndarray]:
    """레코드의 배열 키들을 bool + 정상 축 방향으로 변환."""
    out = {}
    for key in ARRAY_KEYS:
        arr = np.asarray(rec[key], dtype=bool)
        arr = orient(arr, array_id, key)
        if arr.shape[0] != batch:
            raise IngestError(
                f"{array_id}/{key}: batch 크기 불일치. X={batch}, 배열={arr.shape[0]}"
            )
        out[key] = arr
    return out


def build_rows(rec: dict, array_id: str) -> list[dict]:
    """batch를 풀어 x 하나당 catalog 한 줄."""
    xs = rec[X_KEY]
    rows = []
    for pos, x in enumerate(xs):
        row = {"x": x, "array_id": array_id, "batch_pos": pos}
        for key in LIST_KEYS:
            row[key] = rec[key][pos]
        for key in SCALAR_KEYS:
            row[key] = rec[key]  # batch 공통
        rows.append(row)
    return rows


def check_list_batch(rec: dict, array_id: str, batch: int) -> None:
    for key in LIST_KEYS:
        n = len(rec[key])
        if n != batch:
            raise IngestError(
                f"{array_id}/{key}: batch 크기 불일치. X={batch}, list={n}"
            )


# --------------------------------------------------------------------------
# 파이프라인
# --------------------------------------------------------------------------
def ingest(raw_dir: Path, db_dir: Path) -> dict:
    catalog_path = db_dir / CATALOG_NAME
    arrays_dir = db_dir / ARRAYS_DIRNAME
    arrays_dir.mkdir(parents=True, exist_ok=True)

    timing = {"read": 0.0, "normalize": 0.0, "npz": 0.0, "catalog": 0.0}
    t0 = time.perf_counter()
    done = load_done_ids(catalog_path)
    timing["read"] += time.perf_counter() - t0

    files = sorted(raw_dir.glob("*.jsonl"))
    n_new, n_rows, n_skipped_done, n_skipped_incomplete = 0, 0, 0, 0

    with catalog_path.open("ab") as catalog:
        for path in files:
            t = time.perf_counter()
            records = read_run(path)
            timing["read"] += time.perf_counter() - t

            for iteration in sorted(records):
                rec = records[iteration]
                # run마다 iteration이 1부터 리셋되므로 (파일명, iteration)이 고유 키.
                array_id = f"{path.stem}:{iteration}"

                if array_id in done:
                    n_skipped_done += 1
                    continue
                missing = [k for k in REQUIRED_KEYS if k not in rec]
                if missing:
                    # 아직 기록 중인 iteration. 다음 실행에 맡긴다.
                    n_skipped_incomplete += 1
                    continue

                t = time.perf_counter()
                batch = len(rec[X_KEY])
                check_list_batch(rec, array_id, batch)
                arrays = build_arrays(rec, array_id, batch)
                rows = build_rows(rec, array_id)
                timing["normalize"] += time.perf_counter() - t

                # npz 먼저, catalog 나중. catalog가 완료 표시다.
                t = time.perf_counter()
                np.savez_compressed(arrays_dir / f"{array_id}.npz", **arrays)
                timing["npz"] += time.perf_counter() - t

                t = time.perf_counter()
                for row in rows:
                    catalog.write(orjson.dumps(row))
                    catalog.write(b"\n")
                catalog.flush()
                timing["catalog"] += time.perf_counter() - t

                done.add(array_id)
                n_new += 1
                n_rows += len(rows)

    return {
        "files": len(files),
        "new_arrays": n_new,
        "new_rows": n_rows,
        "skipped_done": n_skipped_done,
        "skipped_incomplete": n_skipped_incomplete,
        "timing": timing,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="raw/*.jsonl -> database/")
    ap.add_argument("--raw-dir", default="raw", type=Path)
    ap.add_argument("--db-dir", default="database", type=Path)
    args = ap.parse_args(argv)

    if not args.raw_dir.is_dir():
        print(f"raw 디렉터리 없음: {args.raw_dir}", file=sys.stderr)
        return 1

    started = time.perf_counter()
    try:
        stats = ingest(args.raw_dir, args.db_dir)
    except IngestError as e:
        print(f"[ingest] 중단: {e}", file=sys.stderr)
        return 1
    total = time.perf_counter() - started

    print(
        f"[ingest] 파일 {stats['files']}개 / 새 array {stats['new_arrays']}개 / "
        f"새 catalog 줄 {stats['new_rows']}개 / "
        f"스킵(기존) {stats['skipped_done']} / "
        f"스킵(미완성) {stats['skipped_incomplete']}"
    )
    for stage, seconds in stats["timing"].items():
        print(f"[ingest] {stage:<10} {seconds:7.3f}s")
    print(f"[ingest] {'total':<10} {total:7.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
