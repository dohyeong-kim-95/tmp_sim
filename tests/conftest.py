import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.make_fixture import code_to_ord_stub, make_fixture  # noqa: E402
from util._update_database import _update_database  # noqa: E402


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
    """합성 raw로 갱신한 database/."""
    target = tmp_path / "database"
    _update_database(raw_dir["raw_dir"], target)
    return target
