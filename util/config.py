"""config.toml 로더.

키 이름·차원처럼 실험마다 달라지는 값은 코드에 하드코딩하지 않고 config.toml에
둔다. 이 파일이 그 값들의 유일한 출처다.

튜닝 파라미터(k, trust_dist, min_obs ...)는 여기 들어오지 않는다. 그건 실험이
정해주는 사실이 아니라 탐색 전략의 선택이라 MOCKCalculator.MockConfig에 남는다.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAME = "config.toml"
EXAMPLE_NAME = "config.example.toml"


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    x_key: str
    array_keys: tuple[str, ...]
    list_keys: tuple[str, ...]
    scalar_keys: tuple[str, ...]
    axis1_size: int
    raw_dir: Path
    db_dir: Path

    @property
    def required_keys(self) -> tuple[str, ...]:
        """한 iteration이 완성되었다고 보기 위해 다 모여야 하는 키."""
        return (self.x_key,) + self.array_keys + self.list_keys + self.scalar_keys


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path is not None else ROOT / CONFIG_NAME
    if not path.exists():
        raise ConfigError(
            f"{path}가 없다. {EXAMPLE_NAME}을 복사해 {CONFIG_NAME}을 만들고 실제 "
            f"실험 값으로 고칠 것:  cp {EXAMPLE_NAME} {CONFIG_NAME}"
        )
    with path.open("rb") as f:
        raw = tomllib.load(f)

    try:
        keys, ingest = raw["keys"], raw["ingest"]
        return Config(
            x_key=keys["x"],
            array_keys=tuple(keys["array"]),
            list_keys=tuple(keys["list"]),
            scalar_keys=tuple(keys["scalar"]),
            axis1_size=int(ingest["axis1_size"]),
            raw_dir=Path(ingest["raw_dir"]),
            db_dir=Path(ingest["db_dir"]),
        )
    except KeyError as e:
        raise ConfigError(f"{path}에 {e} 항목이 없다. {EXAMPLE_NAME}과 비교할 것.") from e
