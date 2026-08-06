"""config.toml이 실험 고유 값의 유일한 출처인지 확인한다."""

import numpy as np
import orjson
import pytest

import util.ingest as ingest
from MOCKCalculator import ARRAY_SUM_KEY, MOCKCalculator, MockConfig, objective_key
from tests.conftest import CONFIG_PATH
from tests.make_fixture import make_fixture
from util.config import CONFIG_NAME, EXAMPLE_NAME, ROOT, ConfigError, load_config

EXAMPLE_PATH = ROOT / EXAMPLE_NAME
SOURCE_FILES = [ROOT / "MOCKCalculator.py", ROOT / "util" / "ingest.py", ROOT / "util" / "config.py"]
DOC_FILES = [ROOT / "README.md", ROOT / "database" / "README.md"]


def test_example_config_is_complete():
    cfg = load_config(EXAMPLE_PATH)
    assert cfg.x_key
    assert len(cfg.array_keys) == 2
    assert len(cfg.list_keys) == 2
    assert len(cfg.scalar_keys) == 2
    assert cfg.axis1_size > 0
    assert cfg.raw_dir.name and cfg.db_dir.name
    # 키 이름은 서로 겹치면 안 된다 (병합 시 덮어쓴다).
    assert len(set(cfg.required_keys)) == len(cfg.required_keys)


def test_missing_config_points_at_the_example(tmp_path):
    with pytest.raises(ConfigError) as excinfo:
        load_config(tmp_path / CONFIG_NAME)
    message = str(excinfo.value)
    assert EXAMPLE_NAME in message and CONFIG_NAME in message


def test_incomplete_config_names_the_missing_entry(tmp_path):
    path = tmp_path / CONFIG_NAME
    path.write_text('[keys]\nx = "x"\n\n[ingest]\naxis1_size = 4\nraw_dir = "raw"\ndb_dir = "db"\n')
    with pytest.raises(ConfigError, match="array"):
        load_config(path)


def test_ingest_constants_come_from_the_loader():
    assert ingest.X_KEY == ingest.CONFIG.x_key
    assert ingest.ARRAY_KEYS == ingest.CONFIG.array_keys
    assert ingest.LIST_KEYS == ingest.CONFIG.list_keys
    assert ingest.SCALAR_KEYS == ingest.CONFIG.scalar_keys
    assert ingest.AXIS1_SIZE == ingest.CONFIG.axis1_size
    assert ingest.REQUIRED_KEYS == ingest.CONFIG.required_keys


def test_source_does_not_hardcode_key_names():
    """키 이름이 소스에 복제되면 config.toml을 고쳐도 따라오지 않는다."""
    names = ingest.ARRAY_KEYS + ingest.LIST_KEYS + ingest.SCALAR_KEYS
    for path in SOURCE_FILES:
        text = path.read_text()
        for name in names:
            assert name not in text, f"{path.name}에 키 이름 {name!r}이 하드코딩됨"


def test_docs_do_not_duplicate_key_names():
    """문서는 역할 표기만 쓴다. 구체 이름의 예시는 catalog_example.jsonl 하나뿐."""
    example = load_config(EXAMPLE_PATH)
    names = example.array_keys + example.list_keys + example.scalar_keys
    for path in DOC_FILES:
        text = path.read_text()
        for name in names:
            assert name not in text, f"{path}에 키 이름 {name!r}이 복제됨"


def test_test_session_runs_on_the_example_config():
    if CONFIG_PATH.read_bytes() != EXAMPLE_PATH.read_bytes():
        pytest.skip("로컬 config.toml이 예시와 다르다 (실제 실험 값으로 커스터마이즈됨)")
    assert ingest.CONFIG == load_config(EXAMPLE_PATH)


def test_pipeline_runs_end_to_end_on_configured_keys(tmp_path, code_to_ord):
    """설정 파일이 정한 키 이름만으로 fixture 생성 -> ingest -> MOCK이 전부 돈다."""
    cfg = ingest.CONFIG
    info = make_fixture(tmp_path / "raw")

    # raw 로그가 설정된 키 이름으로 쓰였는지.
    first = orjson.loads((tmp_path / "raw" / "run_a.jsonl").read_bytes().splitlines()[0])
    record = next(iter(first.values()))
    assert set(record) == set(cfg.required_keys)

    stats = ingest.ingest(tmp_path / "raw", tmp_path / "database")
    assert stats["new_arrays"] == len(info["complete_array_ids"])

    rows = [
        orjson.loads(line)
        for line in (tmp_path / "database" / ingest.CATALOG_NAME).read_bytes().splitlines()
        if line.strip()
    ]
    expected = {"x", "array_id", "batch_pos"} | set(cfg.list_keys) | set(cfg.scalar_keys)
    assert all(set(row) == expected for row in rows)

    npz_path = tmp_path / "database" / ingest.ARRAYS_DIRNAME / f"{rows[0]['array_id']}.npz"
    with np.load(npz_path) as npz:
        assert set(npz.files) == set(cfg.array_keys)
        assert npz[cfg.array_keys[0]].shape[2] == cfg.axis1_size

    mock = MOCKCalculator(tmp_path / "database", MockConfig(k=3), code_to_ord).fit()
    result = mock.predict(rows[0]["x"])
    assert set(result.objectives) == {ARRAY_SUM_KEY} | {objective_key(k) for k in cfg.list_keys}
    assert set(result.scalars) == set(cfg.scalar_keys)
    assert mock.self_check()["pass_rate"] == 1.0
