# database/ 스키마

`util/ingest.py`가 `raw/*.jsonl`로부터 만드는 산출물. 이 디렉터리의 내용물은
언제든 `raw/`에서 재생성 가능한 파생물이므로 이 문서 외에는 버전 관리하지 않는다.

```
database/
  README.md            이 문서 (유일하게 커밋되는 파일)
  catalog.jsonl        x 하나당 한 줄
  arrays/{array_id}.npz  5D bool 배열을 batch째로 저장
```

키 이름(`yARR1_key` 등)과 `AXIS1_SIZE`는 **플레이스홀더**다. 실제 값은
`util/ingest.py` 상단 상수를 참조·수정한다.

## catalog.jsonl

JSON Lines. 한 줄이 **x 하나의 한 번의 관측**을 나타낸다.

| 키 | 타입 | 의미 |
| --- | --- | --- |
| `x` | string | 입력 code. `optimizer.code_to_ord`로 ordinal 정수 리스트가 된다 |
| `array_id` | string | `"{raw 파일 stem}:{iteration}"`. npz 파일명이자 배열 좌표의 앞부분 |
| `batch_pos` | int | 그 npz 배열의 batch 축 인덱스. `0 <= batch_pos < batch` |
| `yLST1_key` | list[float] | list y. **이 리스트의 평균이 최소화 대상 목적함수** |
| `yLST2_key` | list[float] | list y. 위와 동일 |
| `y1_cfg_key` | float | batch 공통 설정값. 목적함수가 아니다 |
| `y2_cfg_key` | float | batch 공통 설정값. 목적함수가 아니다 |

실제 예시는
[`tests/fixture_output/catalog_example.jsonl`](../tests/fixture_output/catalog_example.jsonl)
— 손으로 쓴 게 아니라 합성 fixture를 ingest한 산출물에서 5줄을 뽑은 것이다.
`x = "AAA"`가 `run_a:1`과 `run_b:1`에 각각 나타나는 반복 관측이 들어 있고,
`run_a:1`을 공유하는 두 줄의 `y1_cfg_key`가 같은 것도 볼 수 있다.

```json
{"x": "AAA", "array_id": "run_a:1", "batch_pos": 0,
 "yLST1_key": [2.961622, 2.957291, 2.955883],
 "yLST2_key": [3.946742, 3.940358, 3.961977],
 "y1_cfg_key": 9.98714, "y2_cfg_key": 11.000407}
```

성질:

- **같은 `x`가 여러 줄에 나타난다.** TRUE_CALCULATOR는 noisy하므로 같은 x를
  여러 번 관측한 것이고, 그 반복 관측이 곧 서로게이트의 재료다.
- `array_id`는 run(파일)마다 iteration이 1부터 리셋되기 때문에 필요하다.
  `(파일명, iteration)`이 고유 키다.
- 같은 `array_id`를 가진 줄들은 `y1_cfg_key`/`y2_cfg_key`가 서로 같다(batch 공통).
- 줄 순서에 의미는 없다. 파일은 append-only이며, 새 run이 ingest되면 뒤에 붙는다.
- `array_id`가 catalog에 있다는 것이 **그 iteration의 처리 완료 표시**다.
  ingest는 npz를 먼저 쓰고 catalog를 나중에 쓴다.

## arrays/{array_id}.npz

`np.savez_compressed`로 저장된다. 배열 키는 `ARRAY_KEYS`와 같다.

| npz 키 | dtype | shape |
| --- | --- | --- |
| `yARR1_key` | `bool` | `(batch, i0, i1, i2, i3, i4)` |
| `yARR2_key` | `bool` | `(batch, i0, i1, i2, i3, i4)` |

축 의미:

- **축 0 = batch.** catalog의 `batch_pos`가 이 축의 인덱스다.
- **축 1~5 = 실험 배열의 내부 5D 축**(`i0`~`i4`). 내용은 볼록한 blob 형태다.
- **`i1`(= npz 배열의 축 2)은 고정 크기 `AXIS1_SIZE`를 갖는다.** 일부 run은
  내부 축 0과 1이 뒤바뀐 채 기록되는데, ingest가 이 고정 크기로 판정해
  `swapaxes(1, 2)`로 통일한다. 즉 **database에 들어온 배열은 항상 정방향이다.**
  판정이 불가능하면(축 0과 1이 둘 다 `AXIS1_SIZE`이거나 둘 다 아니면) ingest는
  조용히 통과시키지 않고 중단한다.

x 단위로 쪼개지 않는 이유는 파일 수 폭발을 피하기 위해서다.

## 참조 규약

catalog의 `(array_id, batch_pos)`가 배열 좌표다.

```python
import numpy as np, orjson

row = orjson.loads(open("database/catalog.jsonl", "rb").readline())
with np.load(f"database/arrays/{row['array_id']}.npz") as npz:
    y1 = npz["yARR1_key"][row["batch_pos"]]   # 이 x의 5D bool 배열 한 번의 관측
```

같은 x의 반복 관측을 모으려면 catalog에서 `x`로 묶은 뒤 각 줄의 좌표로 배열을
꺼내 쌓는다(`MOCKCalculator.fit()`이 하는 일).
