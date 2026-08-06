import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from util.config import CONFIG_NAME, EXAMPLE_NAME  # noqa: E402

# util.ingest는 import 시점에 config.toml을 읽는다. 신선한 체크아웃이나 CI에는 그
# 파일이 없으므로, 문서에 적힌 그대로 예시를 복사해 만든다. 즉 테스트는
# config.example.toml이 정한 키 이름 위에서 돈다. 우리가 만든 경우에만 지운다.
CONFIG_PATH = ROOT / CONFIG_NAME
_CREATED_CONFIG = not CONFIG_PATH.exists()
if _CREATED_CONFIG:
    shutil.copy(ROOT / EXAMPLE_NAME, CONFIG_PATH)

from tests.make_fixture import code_to_ord_stub, make_fixture  # noqa: E402
from util.ingest import ingest  # noqa: E402


def pytest_sessionfinish(session, exitstatus):
    if _CREATED_CONFIG and CONFIG_PATH.exists():
        CONFIG_PATH.unlink()


@pytest.fixture
def code_to_ord():
    """optimizer.py가 없으므로 테스트는 stub을 주입한다."""
    return code_to_ord_stub


@pytest.fixture
def raw_dir(tmp_path):
    """합성 raw/*.jsonl과 그 요약."""
    info = make_fixture(tmp_path / "raw")
    return info


@pytest.fixture
def db_dir(raw_dir, tmp_path):
    """합성 raw를 ingest한 database/."""
    target = tmp_path / "database"
    ingest(raw_dir["raw_dir"], target)
    return target
