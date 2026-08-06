"""util/_update_database.py 헤더가 실험 고유 값의 유일한 출처인지 확인한다."""

import numpy as np
import orjson

import util._update_database as udb
from MOCKCalculator import ARRAY_SUM_KEY, MOCKCalculator, MockConfig, objective_key
from tests.conftest import ROOT
from tests.make_fixture import make_fixture

# 상수를 소유한 모듈은 당연히 키 이름을 담고 있으므로 제외한다.
SOURCE_FILES = [ROOT / "MOCKCalculator.py"]
DOC_FILES = [ROOT / "README.md", ROOT / "database" / "README.md"]


def test_header_constants_are_complete():
    assert udb.X_KEY
    assert len(udb.ARRAY_KEYS) == 2
    assert len(udb.LIST_KEYS) == 2
    assert len(udb.SCALAR_KEYS) == 2
    assert udb.AXIS1_SIZE > 0
    assert udb.RAW_DIR.name and udb.DB_DIR.name
    # 키 이름은 서로 겹치면 안 된다 (병합 시 덮어쓴다).
    assert len(set(udb.REQUIRED_KEYS)) == len(udb.REQUIRED_KEYS)


def test_required_keys_cover_every_declared_key():
    """필수 키 목록이 헤더의 키 선언에서 파생되는지 — 손으로 나열하면 어긋난다."""
    assert set(udb.REQUIRED_KEYS) == (
        {udb.X_KEY} | set(udb.ARRAY_KEYS) | set(udb.LIST_KEYS) | set(udb.SCALAR_KEYS)
    )


def test_source_does_not_hardcode_key_names():
    """키 이름이 다른 소스에 복제되면 헤더를 고쳐도 따라오지 않는다."""
    names = udb.ARRAY_KEYS + udb.LIST_KEYS + udb.SCALAR_KEYS
    for path in SOURCE_FILES:
        text = path.read_text()
        for name in names:
            assert name not in text, f"{path.name}에 키 이름 {name!r}이 하드코딩됨"


def test_docs_do_not_duplicate_key_names():
    """문서는 역할 표기만 쓴다. 구체 이름의 예시는 catalog_example.jsonl 하나뿐."""
    names = udb.ARRAY_KEYS + udb.LIST_KEYS + udb.SCALAR_KEYS
    for path in DOC_FILES:
        text = path.read_text()
        for name in names:
            assert name not in text, f"{path}에 키 이름 {name!r}이 복제됨"


def test_pipeline_runs_end_to_end_on_declared_keys(tmp_path, code_to_ord):
    """헤더가 정한 키 이름만으로 fixture 생성 -> database 갱신 -> MOCK이 전부 돈다."""
    info = make_fixture(tmp_path / "raw")

    # raw 로그가 선언된 키 이름으로 쓰였는지.
    first = orjson.loads((tmp_path / "raw" / "run_a.jsonl").read_bytes().splitlines()[0])
    record = next(iter(first.values()))
    assert set(record) == set(udb.REQUIRED_KEYS)

    stats = udb._update_database(tmp_path / "raw", tmp_path / "database")
    assert stats["new_arrays"] == len(info["complete_array_ids"])

    rows = [
        orjson.loads(line)
        for line in (tmp_path / "database" / udb.CATALOG_NAME).read_bytes().splitlines()
        if line.strip()
    ]
    expected = {"x", "array_id", "batch_pos"} | set(udb.LIST_KEYS) | set(udb.SCALAR_KEYS)
    assert all(set(row) == expected for row in rows)

    npz_path = tmp_path / "database" / udb.ARRAYS_DIRNAME / f"{rows[0]['array_id']}.npz"
    with np.load(npz_path) as npz:
        assert set(npz.files) == set(udb.ARRAY_KEYS)
        assert npz[udb.ARRAY_KEYS[0]].shape[2] == udb.AXIS1_SIZE

    mock = MOCKCalculator(tmp_path / "database", MockConfig(k=3), code_to_ord).fit()
    result = mock.predict(rows[0]["x"])
    assert set(result.objectives) == {ARRAY_SUM_KEY} | {
        objective_key(k) for k in udb.LIST_KEYS
    }
    assert set(result.scalars) == set(udb.SCALAR_KEYS)
    assert mock.self_check()["pass_rate"] == 1.0
