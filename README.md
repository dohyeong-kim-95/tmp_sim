# tmp_sim

비싸고 noisy한 실험 평가기(이하 **TRUE_CALCULATOR**)를 대상으로 하는 최적화 작업의
보조 부품 두 개를 담는 저장소.

## 1. 배경

- 입력 `X`는 string code. `optimizer.py`의 `code_to_ord` / `ord_to_code`로
  ordinal 정수 리스트와 상호 변환된다.
- TRUE_CALCULATOR의 출력은 batch 단위로:
  - bool 5D array 2개 (내용은 볼록한 blob 형태)
  - list 2개 — **각 x의 list의 평균이 최소화 대상 목적함수**
  - scalar 2개 — batch 공통 설정값. **목적함수가 아니다**
- 최종 목표는 **bool array의 합이 크고 list y의 평균이 작은 X**를 GA/SA 계열로
  찾는 것. 즉 다목적(multi-objective)이다.
- 같은 X라도 실행마다 출력이 조금씩 다르다(noise). 따라서 한 X에 대한 반복 관측을
  모으는 것 자체가 데이터 자산이다.

TRUE_CALCULATOR 호출은 비싸므로, 축적된 실행 로그로 TRUE를 흉내내는 서로게이트를
두고 후보 대부분을 서로게이트로 걸러낸다. 이 저장소는 그 두 부품을 담는다.

- **A. database 갱신 파이프라인** (`util/_update_database.py`): `raw/*.jsonl` → `database/`
- **B. 서로게이트** (`MOCKCalculator.py`): `database/`로 TRUE를 흉내냄

`optimizer.py`(GA/SA 본체, `code_to_ord`/`ord_to_code`)와 TRUE_CALCULATOR 자체는
이 저장소 밖에 있으며, 여기서는 **import만 한다**.

## 2. 구조

```
README.md
.gitignore
util/
  _update_database.py  # raw/*.jsonl -> database/ (A). 실험 고유 상수도 이 파일 헤더에 있다
MOCKCalculator.py      # 서로게이트 (B)
tests/
  make_fixture.py      # 스펙만으로 합성 raw/*.jsonl 생성
  conftest.py
  test_update_database.py
  test_constants.py
  test_mock.py
  test_validation_contract.py
  fixture_output/
    catalog_example.jsonl  # database 갱신 산출물에서 뽑은 catalog 예시 5줄

raw/                 # (gitignore) TRUE_CALCULATOR 실행 로그, run 하나당 파일 하나
database/            # (gitignore, README.md만 예외) 갱신 산출물
  README.md          #   catalog/npz 스키마 문서
  catalog.jsonl      #   x 하나당 한 줄
  arrays/{array_id}.npz
```

`raw/`, `database/`는 데이터라서 버전 관리하지 않는다. `database/`는 언제든
`raw/`로부터 재생성 가능한 파생물이다. 단 스키마 문서
[`database/README.md`](database/README.md)만 `.gitignore` 예외로 커밋된다.

## 3. 설정 (상수 헤더)

실제 실험의 키 이름·차원은 **`util/_update_database.py` 파일 맨 위 상수 블록**에
모여 있다. 설정 파일도 로더도 없다 — 그 블록이 유일한 출처이고, 다른 모듈은
거기서 import해서 쓴다.

```python
from util._update_database import ARRAY_KEYS, LIST_KEYS, SCALAR_KEYS
```

실험이 바뀌면 그 블록만 고친다. 담기는 값:

| 상수 | 의미 |
| --- | --- |
| `X_KEY` | X(코드 문자열 batch)의 키 이름 |
| `ARRAY_KEYS` | bool 5D array 2개의 키 이름 |
| `LIST_KEYS` | list y 2개의 키 이름 (평균이 목적함수) |
| `SCALAR_KEYS` | batch 공통 설정값 2개의 키 이름 |
| `AXIS1_SIZE` | 배열 내부 축1의 고정 크기 (축 뒤바뀜 판정용) |
| `RAW_DIR`, `DB_DIR` | 입출력 디렉터리 |
| `REQUIRED_KEYS` | 위 키 선언에서 파생 — 한 iteration이 완성되었다고 보는 기준 |

커밋된 값은 자리를 잡아두기 위한 **플레이스홀더**이며 실제 실험의 키 이름·차원이
아니다. 실제 로그에 맞춰 고쳐 쓴다.

서로게이트의 **튜닝 파라미터는 여기 들어가지 않는다.** `k`, `trust_dist`,
`min_obs`, `dilate_d`, `z`, `abs_tol`은 실험이 정해주는 사실이 아니라 탐색 전략의
선택이라 `MOCKCalculator.MockConfig` dataclass에 남는다.

아래 문서에서는 구체 키 이름 대신 **역할 표기**(`<array key 1>`, `<list key 1>`,
`<cfg key 1>` …)를 쓴다. 실제 이름은 상수 블록을 보면 된다.

## 4. A. database 갱신 파이프라인

### 입력 포맷 (`raw/*.jsonl`)

한 줄 = `{"<iteration>": {"<키>": <값>}}`. 한 iteration의 정보가 **두 줄**
(배열 줄 + 스칼라 줄)에 나뉘어 기록되므로, 같은 iteration의 줄들을 **null이 아닌
값으로 병합**해서 하나의 레코드로 만든다.

```jsonl
{"1": {"<x key>": ["A", "B"], "<array key 1>": [[...]], "<list key 1>": [[...]], "<cfg key 1>": null}}
{"1": {"<x key>": null,       "<array key 1>": null,    "<list key 1>": null,    "<cfg key 1>": 0.5}}
```

run(파일)마다 iteration 번호가 1부터 리셋되므로 고유 키는 **(파일명, iteration)**.
`array_id = "{파일 stem}:{iteration}"`으로 명명한다.

### batch 정렬 규약

`X`와 배열/리스트 y의 첫 축은 batch로 정렬되어 있다. 즉 `X[i]`의 결과가 `y[i]`.
스칼라 y는 batch 공통 설정값이므로 batch 내 모든 x가 같은 값을 갖는다.

### 축 뒤바뀜 정규화

일부 run은 배열 **내부 축 0과 1이 뒤바뀌어** 기록된다. 내부 축1이 고정 크기
(`AXIS1_SIZE`)를 갖는 점을 이용해 판정한다. batch 축을 포함한 기록 배열은 6D
`(batch, i0, i1, i2, i3, i4)`이고,

- `shape[2] == AXIS1_SIZE` 이고 `shape[1] != AXIS1_SIZE` → 정상
- `shape[1] == AXIS1_SIZE` 이고 `shape[2] != AXIS1_SIZE` → 뒤바뀜, `swapaxes(1, 2)`
- 그 외(둘 다 같거나 둘 다 다름) → **판정 불가, 에러로 중단**

조용히 통과시키면 데이터베이스가 오염되므로 중단이 맞다.

### 출력 1: `database/catalog.jsonl`

batch를 풀어 **x 하나당 한 줄**.

```json
{"x": "<code>", "array_id": "run_a:1", "batch_pos": 0,
 "<list key 1>": [...], "<list key 2>": [...], "<cfg key 1>": 0.5, "<cfg key 2>": 1.2}
```

같은 x가 여러 번 관측되면 줄이 여러 개 생긴다. 이것이 곧 **noise 반복 관측
수집**이며, 서로게이트 층1과 검증 계약의 재료다.

list y는 평균만 쓰이지만 catalog에는 **원본 리스트 그대로** 저장한다. database는
손실 없이 보관하고, 목적함수로의 파생(mean)은 `MOCKCalculator`가 한다.

한 줄의 전체 스키마는 [`database/README.md`](database/README.md) 참고. 실제 줄이
어떻게 생겼는지는 합성 fixture로 database를 갱신해 뽑은
[`tests/fixture_output/catalog_example.jsonl`](tests/fixture_output/catalog_example.jsonl)
을 보면 된다(반복 관측된 x 포함).

### 출력 2: `database/arrays/{array_id}.npz`

5D bool 배열을 **batch째로** `bool` dtype + `savez_compressed`로 저장한다
(키는 `ARRAY_KEYS` 그대로). x 단위로 쪼개면 파일 수가 폭발하므로 쪼개지 않는다.
catalog의 `(array_id, batch_pos)`가 배열 안의 좌표 역할을 한다.

### 멱등성

- catalog에 이미 있는 `array_id`는 스킵한다.
- 필수 키가 다 모이지 않은 iteration(기록 도중이라 줄이 덜 쌓인 경우)도 스킵하고
  다음 실행에 맡긴다.
- npz를 먼저 쓰고 catalog를 나중에 쓴다. catalog가 완료 표시이므로, 중간에 죽어도
  다시 실행하면 그 iteration부터 다시 처리된다.

### torn write 내성

`raw/*.jsonl`은 실험이 실시간으로 append하는 파일이라, 갱신이 도는 시점에
**마지막 줄이 쓰다 만 상태**일 수 있다. 이건 손상이 아니라 정상 상황이다.

- 파일의 **마지막 줄** 파싱 실패 → 경고만 찍고 스킵. 다음 실행에서 완성된 줄을 읽는다.
- **마지막이 아닌 줄** 파싱 실패 → 진짜 손상이므로 에러로 중단.

### 실행

```bash
python util/_update_database.py                      # raw/ -> database/
python util/_update_database.py --raw-dir raw --db-dir database
```

단계별 소요 시간(파싱 / 정규화 / npz 쓰기 / catalog 쓰기)을 마지막에 출력한다.
JSON 파싱은 `orjson`을 쓴다(성능상 필수로 검증됨).

## 5. B. MOCKCalculator

`fit()`이 `database/`를 통째로 메모리에 올리는 **스냅샷** 방식. 새 데이터가
들어오면 `_update_database`를 다시 돌리고 `fit()`을 다시 부른다.

적재와 모델링은 갈라져 있다.

- `_load_database()` — catalog와 npz를 읽어 x별 `RawObservations`(관측 배열
  리스트 + 그 x의 catalog 줄들)로 모은다. **디스크를 만지는 건 여기까지다.**
  같은 npz를 두 번 열지 않도록 catalog를 `array_id`로 묶어 파일당 한 번만 연다.
- `fit()` — 그 결과 위에서 합의 배열, ordinal 행렬, 거리 정규화용 range를
  만든다. 여기서부터는 디스크를 보지 않는다.

### 2층 구조

- **층1 — 관측된 x**: 배열은 반복 관측을 **원소별 다수결**로 합쳐
  합의(consensus)를 반환한다. `p_arrays`는 관측 중 True 비율.
  list y는 원소 단위로 합의하지 않는다 — 관측마다 `mean(list)`를 구해 스칼라처럼
  다루고, 그 값들의 평균이 층1의 예측이다.
- **층2 — 미관측 x**: ordinal 벡터 공간의 **정규화 L1 거리**
  `sum(|a-b| / range) / n_vars`로 k최근접 관측을 찾아 **거리 가중 soft 평균** →
  `p_arrays`(원소별 [0,1]). `p >= 0.5`가 bool 예측.
  list y의 평균값과 설정 스칼라도 같은 가중치로 평균낸다.
  최근접 거리가 `trust_dist`를 넘으면 `needs_test=True`
  ("서로게이트를 믿지 말고 TRUE로 실제 평가하라"는 신호).

### 목적함수 — MOCK은 스칼라화하지 않는다

목적은 **다목적**이다. `MockResult.objectives`가 목적별 값을 그대로 노출한다.

| 이름 | 방향 | 뜻 |
| --- | --- | --- |
| `array_sum` | 최대화 | `p_arrays`의 합. bool로 반올림한 합보다 gradient-free 탐색에서 신호가 매끄럽다 |
| `<list key 1>_mean` | 최소화 | 그 x의 첫 번째 list y의 평균 |
| `<list key 2>_mean` | 최소화 | 그 x의 두 번째 list y의 평균 |

목적함수 이름은 `objective_key()`가 `LIST_KEYS`에서 파생시킨다. 상수 블록의
키 이름을 바꾸면 objectives 이름도 따라 바뀐다.

**가중 결합·스칼라화는 MOCK의 일이 아니라 옵티마이저의 책임이다.** trade-off를
어떻게 고를지(가중합, Pareto front, constraint 처리)는 탐색 전략의 문제이고,
서로게이트가 임의의 가중치를 박아 넣으면 그 선택이 숨어버린다. 그래서 MOCK에는
단일 `score`도 `evaluate()`도 없다.

`SCALAR_KEYS`의 두 값은 batch 공통 설정값이라 목적함수가 아니다.
`result.scalars`에 따로 담기고, 검증에만 쓰인다.

### 검증 계약

예측이 "관측과 모순되지 않는가"를 판정하는 규칙. 관측이 있는 x에 대해서만 성립한다.

- **배열**
  - 관측 충분(`n_obs >= min_obs`): `관측 교집합 ⊆ 예측 ⊆ 관측 합집합`
  - 부족: 합의 배열의 `erode(d) ⊆ 예측 ⊆ dilate(d)` (거리 `d = dilate_d`, Manhattan)
- **스칼라** — list y 파생 목적함수(`*_mean`)와 설정 스칼라에 같은 규칙을 쓴다
  - 충분: `mean ± z·σ`
  - 부족: `mean ± abs_tol`

파라미터는 전부 `MockConfig` dataclass에 모여 있다.

### 자가 점검

- `self_check()` — 각 관측 x의 합의 배열이 자기 관측 범위를 통과하는지.
  구조적으로 항상 통과해야 하며, 실패하면 database 갱신이나 합의 로직이 깨진 것이다.
- `loo_check()` — leave-one-out. 어떤 x를 이웃 풀에서 빼고 층2로 예측한 뒤 그 x의
  실제 관측으로 검증한다. 층2의 예측력과 `trust_dist` 설정의 타당성을 본다.

### 사용

```python
from MOCKCalculator import MOCKCalculator, MockConfig

mock = MOCKCalculator(config=MockConfig(k=5, trust_dist=0.05))   # db_dir 기본값은 DB_DIR
mock.fit()

r = mock.predict(candidate_code)
r.objectives   # {"array_sum": ..., "<list key 1>_mean": ..., "<list key 2>_mean": ...}
r.needs_test   # True면 TRUE_CALCULATOR로 실제 평가

print(mock.self_check())
print(mock.loo_check())
```

`code_to_ord`는 기본적으로 `optimizer`에서 **늦게(lazy) import**된다. 모듈을
불러오는 것만으로는 optimizer를 요구하지 않고, ordinal이 실제로 필요한 시점에만
찾는다. 테스트처럼 optimizer가 없는 환경에서는 생성자로 주입한다.

```python
mock = MOCKCalculator(code_to_ord=lambda code: [ord(c) for c in code])
```

CLI로도 점검할 수 있다(이쪽은 `optimizer.py`가 필요하다).

```bash
python MOCKCalculator.py --db-dir database   # fit + self_check + loo_check 요약
```

## 6. 운영 사이클

```
1. 옵티마이저가 후보 X들을 MOCKCalculator로 평가
2. needs_test=True인 후보만 TRUE_CALCULATOR로 실제 평가
3. TRUE 실행 로그가 raw/에 쌓임
4. python util/_update_database.py  (멱등 — 새 run만 처리)
5. mock.fit() 재호출 → 서로게이트가 넓어진 관측 영역을 반영
6. 1로
```

반복할수록 관측 영역이 넓어져 `needs_test` 비율이 떨어지고, 비싼 TRUE 호출이
줄어든다.

## 7. 테스트

실제 데이터 없이 돈다. `tests/make_fixture.py`가 **스펙만으로** 소형 합성
`raw/*.jsonl`을 만들고, 그걸로 database를 갱신해 두 부품을 검증한다.

```bash
pip install pytest
python -m pytest tests -q
```

fixture가 담는 케이스:

| 케이스 | 어디에 |
| --- | --- |
| 두 줄 분할 기록 (배열 줄 + 스칼라 줄) | 모든 run |
| 파일 간 iteration 번호 리셋 | `run_a`~`run_d` 전부 iteration 1부터 |
| 같은 x의 반복 관측 (blob 경계 1~2원소 흔들림) | `run_a` + `run_b`가 같은 코드 관측 |
| 축 0/1이 뒤바뀐 run | `run_b` |
| 미완성 iteration | `run_c:3` (배열 줄만) |
| 쓰다 만 마지막 줄 | `run_d` 끝 |

fixture만 따로 만들어 눈으로 볼 수도 있다.

```bash
python tests/make_fixture.py raw
```

`tests/fixture_output/catalog_example.jsonl`은 그 fixture로 database를 갱신한
결과에서 5줄을 뽑아 커밋한 것이다. 스키마 문서의 예시로 쓰이며, 키 이름이
바뀌었는데 예시가 따라오지 않으면 테스트가 실패한다. 구체 키 이름이 적힌 곳은 이
파일과 `util/_update_database.py`의 상수 블록뿐이고, 다른 소스나 문서에 이름이
복제되면 `test_constants.py`가 잡는다.

`optimizer.py`가 없으므로 테스트는 `code_to_ord` stub(문자→ordinal)을 생성자로
주입한다. 즉 테스트는 상수 블록이 정한 키 이름 위에서 돌고,
`test_constants.py`가 그 사실을 확인한다.

## 8. 의존성

Python 3.9 이상. 표준 라이브러리 외의 설정 포맷 의존성은 없다 — 실험 고유 값은
`util/_update_database.py` 헤더의 상수로 들어간다.

```bash
pip install numpy orjson        # 런타임
pip install pytest              # 테스트
```

`optimizer.py`(`code_to_ord`, `ord_to_code`)는 이 저장소에 없다.
`MOCKCalculator.py`는 `code_to_ord`만 필요하고 그것도 늦게 import하므로, 모듈을
불러오거나 stub을 주입하는 데는 optimizer가 없어도 된다. 기본 경로로 쓰려면
`optimizer.py`가 import 경로에 있어야 한다.
`util/_update_database.py`는 optimizer에 의존하지 않으므로 단독 실행 가능하다.
