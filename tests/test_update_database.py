from pathlib import Path

import numpy as np
import orjson
import pytest

from tests.make_fixture import complete_incomplete_iteration, complete_torn_line
from util._update_database import (
    ARRAY_KEYS,
    AXIS1_SIZE,
    CATALOG_NAME,
    ARRAYS_DIRNAME,
    LIST_KEYS,
    SCALAR_KEYS,
    X_KEY,
    UpdateDatabaseError,
    _update_database,
    build_arrays,
    check_list_batch,
    orient,
    read_run,
)


def read_catalog(db_dir):
    with (db_dir / CATALOG_NAME).open("rb") as f:
        return [orjson.loads(line) for line in f if line.strip()]


def test_catalog_has_one_row_per_x(raw_dir, db_dir):
    rows = read_catalog(db_dir)
    expected = len(raw_dir["complete_array_ids"]) * raw_dir["batch"]
    assert len(rows) == expected
    assert {r["array_id"] for r in rows} == set(raw_dir["complete_array_ids"])

    row = rows[0]
    assert set(row) == {"x", "array_id", "batch_pos"} | set(LIST_KEYS) | set(SCALAR_KEYS)
    assert isinstance(row["x"], str)
    assert isinstance(row["batch_pos"], int)
    for key in LIST_KEYS:
        assert isinstance(row[key], list) and row[key]
    for key in SCALAR_KEYS:
        assert isinstance(row[key], float)


def test_scalar_is_shared_across_batch(db_dir):
    rows = read_catalog(db_dir)
    by_array = {}
    for row in rows:
        by_array.setdefault(row["array_id"], []).append(row)
    for group in by_array.values():
        for key in SCALAR_KEYS:
            assert len({r[key] for r in group}) == 1


def test_npz_shape_dtype_and_axis_normalized(raw_dir, db_dir):
    shape = raw_dir["shape"]
    for array_id in raw_dir["complete_array_ids"]:
        with np.load(db_dir / ARRAYS_DIRNAME / f"{array_id}.npz") as npz:
            assert set(npz.files) == set(ARRAY_KEYS)
            for key in ARRAY_KEYS:
                arr = npz[key]
                assert arr.dtype == np.bool_
                # 축 뒤바뀐 run(run_b)도 같은 방향으로 통일되어야 한다.
                assert arr.shape == (raw_dir["batch"],) + shape
                assert arr.shape[2] == AXIS1_SIZE


def test_catalog_coordinates_point_into_npz(db_dir):
    rows = read_catalog(db_dir)
    for row in rows:
        with np.load(db_dir / ARRAYS_DIRNAME / f"{row['array_id']}.npz") as npz:
            for key in ARRAY_KEYS:
                assert 0 <= row["batch_pos"] < npz[key].shape[0]


def test_update_database_is_idempotent(raw_dir, db_dir):
    before = read_catalog(db_dir)
    stats = _update_database(raw_dir["raw_dir"], db_dir)
    assert stats["new_arrays"] == 0
    assert stats["new_rows"] == 0
    assert stats["skipped_done"] == len(raw_dir["complete_array_ids"])
    assert read_catalog(db_dir) == before


def test_incomplete_iteration_skipped_then_picked_up(raw_dir, db_dir):
    rows = read_catalog(db_dir)
    assert "run_c:3" not in {r["array_id"] for r in rows}

    complete_incomplete_iteration(raw_dir["raw_dir"])
    stats = _update_database(raw_dir["raw_dir"], db_dir)
    assert stats["new_arrays"] == 1
    assert stats["skipped_incomplete"] == 0
    assert "run_c:3" in {r["array_id"] for r in read_catalog(db_dir)}


def test_torn_last_line_warns_and_skips(raw_dir, db_dir, capsys):
    # 쓰다 만 마지막 줄이 있는 run_d도 완성된 iteration은 정상 수집되어야 한다.
    assert "run_d:1" in {r["array_id"] for r in read_catalog(db_dir)}

    read_run(raw_dir["raw_dir"] / raw_dir["torn_file"])
    assert "마지막 줄이 쓰다 만 상태" in capsys.readouterr().err


def test_corrupt_middle_line_aborts(tmp_path, raw_dir):
    path = raw_dir["raw_dir"] / "run_a.jsonl"
    lines = path.read_bytes().splitlines()
    lines[1] = b'{"1": {"x": [broken'          # 마지막이 아닌 줄이 깨짐
    broken = tmp_path / "broken.jsonl"
    broken.write_bytes(b"\n".join(lines) + b"\n")

    with pytest.raises(UpdateDatabaseError, match="JSON 파싱 실패"):
        read_run(broken)


def test_iteration_numbers_reset_per_run(raw_dir, db_dir):
    ids = {r["array_id"] for r in read_catalog(db_dir)}
    # 여러 run이 같은 iteration 번호 1을 쓰지만 array_id로 구분된다.
    assert {"run_a:1", "run_b:1", "run_c:1", "run_d:1"} <= ids


def test_repeated_x_yields_multiple_rows(raw_dir, db_dir):
    rows = read_catalog(db_dir)
    counts = {}
    for row in rows:
        counts[row["x"]] = counts.get(row["x"], 0) + 1
    for code in raw_dir["repeated_codes"]:
        assert counts[code] == 2, code       # run_a + run_b
    for code in raw_dir["single_codes"]:
        assert counts[code] == 1, code


def test_catalog_example_matches_current_schema():
    """커밋된 예시가 스키마 변경(키 개명 등)을 따라오지 못하면 잡아낸다."""
    path = Path(__file__).parent / "fixture_output" / "catalog_example.jsonl"
    rows = [orjson.loads(line) for line in path.read_bytes().splitlines() if line.strip()]

    assert len(rows) == 5
    expected = {"x", "array_id", "batch_pos"} | set(LIST_KEYS) | set(SCALAR_KEYS)
    for row in rows:
        assert set(row) == expected

    # 반복 관측된 x가 들어 있어야 예시로서 값을 한다.
    counts = {}
    for row in rows:
        counts[row["x"]] = counts.get(row["x"], 0) + 1
    assert max(counts.values()) >= 2


def test_torn_line_completed_is_picked_up_on_next_run(raw_dir, db_dir):
    """2회차는 아무 변화 없고, 잘린 줄이 완성된 3회차에 그 iteration만 늘어난다."""
    def npz_count():
        return len(list((db_dir / ARRAYS_DIRNAME).glob("*.npz")))

    rows_before, npz_before = read_catalog(db_dir), npz_count()
    assert "run_d:2" not in {r["array_id"] for r in rows_before}

    stats2 = _update_database(raw_dir["raw_dir"], db_dir)
    assert stats2["new_arrays"] == 0
    assert len(read_catalog(db_dir)) == len(rows_before)
    assert npz_count() == npz_before

    codes = complete_torn_line(raw_dir["raw_dir"])
    stats3 = _update_database(raw_dir["raw_dir"], db_dir)
    assert stats3["new_arrays"] == 1
    assert stats3["new_rows"] == raw_dir["batch"]
    assert npz_count() == npz_before + 1

    rows_after = read_catalog(db_dir)
    assert len(rows_after) == len(rows_before) + raw_dir["batch"]
    added = [r for r in rows_after if r["array_id"] == "run_d:2"]
    assert [r["x"] for r in added] == codes
    # 이미 있던 줄은 그대로다.
    assert rows_after[: len(rows_before)] == rows_before


def test_build_arrays_rejects_batch_mismatch():
    rec = _minimal_record(batch=2)
    with pytest.raises(UpdateDatabaseError, match="batch 크기 불일치"):
        build_arrays(rec, "t:1", batch=3)


def test_build_arrays_rejects_wrong_ndim():
    rec = _minimal_record(batch=2)
    rec[ARRAY_KEYS[0]] = np.zeros((2, 3, AXIS1_SIZE), dtype=bool).tolist()
    with pytest.raises(UpdateDatabaseError, match="6D"):
        build_arrays(rec, "t:1", batch=2)


def test_check_list_batch_rejects_length_mismatch():
    rec = _minimal_record(batch=2)
    with pytest.raises(UpdateDatabaseError, match="batch 크기 불일치"):
        check_list_batch(rec, "t:1", batch=3)


def _minimal_record(batch: int) -> dict:
    """스키마만 맞춘 최소 레코드. 에러 경로 확인용."""
    arr = np.zeros((batch, 3, AXIS1_SIZE, 2, 2, 2), dtype=bool).tolist()
    rec = {X_KEY: ["AAA"] * batch}
    rec.update({key: arr for key in ARRAY_KEYS})
    rec.update({key: [[0.0, 0.0] for _ in range(batch)] for key in LIST_KEYS})
    rec.update({key: 1.0 for key in SCALAR_KEYS})
    return rec


def test_orient_rejects_ambiguous_shapes():
    ok = np.zeros((2, 3, AXIS1_SIZE, 2, 2, 2), dtype=bool)
    assert orient(ok, "t", "k").shape == ok.shape

    swapped = np.zeros((2, AXIS1_SIZE, 3, 2, 2, 2), dtype=bool)
    assert orient(swapped, "t", "k").shape == ok.shape

    for shape in [(2, AXIS1_SIZE, AXIS1_SIZE, 2, 2, 2), (2, 3, 3, 2, 2, 2)]:
        with pytest.raises(UpdateDatabaseError, match="판정 불가"):
            orient(np.zeros(shape, dtype=bool), "t", "k")

    with pytest.raises(UpdateDatabaseError, match="6D"):
        orient(np.zeros((2, 3, AXIS1_SIZE), dtype=bool), "t", "k")
