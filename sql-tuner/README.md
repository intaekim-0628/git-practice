# Oracle SQL Tuner / gpt-oss-120b QLoRA

H200 141GB 1개용 코드 및 재구성 데이터입니다. **현재 클라우드에는 CUDA 장치가 없어 학습은 실행하지 않았습니다.** 실제 GPU 메모리 사용량, 모델 로딩, 토큰 길이, 처리 속도 및 epoch 손실은 아직 측정되지 않았습니다. CPU 데이터 검증과 단위 테스트만 완료했습니다.

## 추가 증강: 현재 기본 학습 데이터

기존 train 39건을 유지하고 합성 변형 552건을 추가하여 `data/train_augmented.jsonl` **591건**을 만들었습니다. eval 22건과 test 13건은 파일 내용과 SHA-256이 그대로입니다. 이전 `data/train.jsonl`도 보존했습니다. 기본 학습 명령은 증강 파일을 사용합니다.

| 증강 유형 | 건수 |
|---|---:|
| 기존 원본 | 39 |
| 동일 근거 + 요청 표현/SQL 렌더링 변형 | 148 |
| 관측 계획 제거 | 92 |
| 인덱스 메타데이터 제거 | 148 |
| 소유자 정보 제거 | 148 |
| SQL만 제공, 계획·메타데이터 제거 | 16 |
| 합계 | 591 |

부모마다 최대 20개의 합성 후보를 만들고 중복을 제거했습니다. SQL 리터럴·테이블·컬럼은 바꾸지 않습니다. AST가 달라지는 렌더링은 실패로 처리합니다. 근거를 제거한 예제에서는 객체 분류와 관측 계획을 다시 만들고 재작성 SQL 권고를 제거합니다. 새로운 성능 수치, 실행 계획 또는 전문가 정답을 만들어 붙이지 않았습니다.

독립적인 SQL 템플릿 계열은 여전히 4개입니다. 591개의 독립적인 튜닝 사례를 확보했다는 뜻은 아닙니다. 근거 부족을 설명하는 예제가 많이 포함되므로 응답이 판단 보류에 치우치는지 eval 생성 결과도 확인해야 합니다. 기존 train을 쓰려면 `--train-file data/train.jsonl`로 지정하세요.

재생성:

```bash
python augment_data.py --data-dir data
python -m unittest discover -p 'test_*.py' -v
```

증강 내역은 `data/augmentation_audit.json`에 있습니다. 증강 파일의 3 epochs는 batch 1, accumulation 8 기준 약 222 optimizer steps입니다. 3~5시간 달성 여부는 실제 H200 처리 속도로 측정해야 합니다.

## 기존 재구성 결과

| 분할 | 레코드 | SQL 템플릿 계열 |
|---|---:|---:|
| train | 39 | 4 |
| eval | 22 | 2 |
| test | 13 | 2 |

원본 `oracle_sql_tuning_corrected_all_v4.jsonl`의 94건을 기준으로 기존 eval·train 파일의 ID와 출처를 연결했습니다. 기존 eval에 속한 SQL 계열은 전체를 eval에 둡니다. test에는 물리 테이블을 조회하는 계열과 비DDL 재작성 계열을 따로 두었습니다. 문자 리터럴/숫자 리터럴을 치환하고 힌트를 제거한 Oracle AST 템플릿으로 그룹을 만들었습니다. 같은 그룹은 분할을 넘지 않습니다. 식별자/별칭을 바꾼 의미상 유사 쿼리까지 완전히 검출했다는 뜻은 아닙니다.

- C등급 12건과 `FROM ( ... hierarchical ... )` 같은 불완전 SQL 1건은 `data/quarantine.jsonl`에 보존했습니다.
- 대소문자·공백 변경만 있는 증강 134건은 중복 가중을 피하려고 제외했습니다.
- 입력과 재구성 정답이 모두 동일한 7건을 합쳐 74건을 사용했습니다.
- 나머지 11개 JSONL의 275건(정확히 다른 레코드 95건)은 `data/legacy_review.jsonl`에 모두 보존했습니다. 관측 계획과 객체 메타데이터가 없어 grounded 정답으로 합치지 않았습니다. SQL 이름/문자열이 비슷하다는 이유로 Oracle 근거를 붙이지 않았습니다.
- `comprehensive_tuning_data.jsonl`은 한 줄의 JSON 배열입니다. 배열 안의 5개 레코드로 정상화해 검사했습니다.
- 모든 업로드 원본은 `sources/`, Oracle 원본 정답과 출처는 `data/oracle_originals.jsonl`에 있습니다.

### 정답 재구성의 의미

**정답은 자동으로 만든 보수적 근거 응답이며 전문가가 검증한 튜닝 정답이 아닙니다.** 현재 테이블 메타데이터에는 `object_type`, `oracle_maintained`가 없습니다. 소유자가 빠진 사례도 있습니다. 그러므로 현 자료만으로 물리 테이블의 DDL 적격성을 입증할 수 없습니다.

74건 중 58건의 원본 답변에서 DDL/힌트 또는 숨겨진 파라미터 권고가 검출됐습니다. 해당 권고는 학습 정답에 넣지 않고 메타데이터 수집 및 원본 유지 응답으로 바꿨습니다. 별도의 SQL 재작성 후보는 16건에만 남았습니다. 이 후보도 결과 동등성을 검증하지 않았으며 실행 전 검토가 필요합니다. `after_plan.verified=true`는 업로드 표시만으로 확인할 수 없어 모두 미측정으로 표현했습니다. `reasoning_targets`의 원본 분석은 사용자 입력에 넣지 않아 정답 누출을 막았습니다.

이 데이터는 **불충분한 근거에서 과도한 권고를 하지 않는 응답 형식 학습**에 가깝습니다. 인덱스 설계/실제 속도 개선 능력을 넓게 학습하려면 전문가가 검증한 사용자 소유 물리 테이블, 인덱스 구성, 원본·튜닝 SQL의 실제 계획과 측정값을 추가해야 합니다. test도 2개 계열에 불과해 일반화 성능을 대표하지 못합니다. 분할 비율을 맞추기 위해 같은 계열을 섞지 않았습니다.

## H200 실행

Linux, CUDA 지원 NVIDIA 드라이버, Python 3.11 또는 3.12 환경에서 압축을 풀고 실행합니다. 모델 파일·캐시·체크포인트를 저장할 충분한 디스크가 필요합니다. 설치 스택은 H200에서 아직 검증하지 않았으며 Unsloth 의존성 해석 결과를 `runtime-lock.txt`에 저장합니다. 모델 다운로드는 공개 Hugging Face 모델을 사용합니다.

```bash
cd sql_tuner_package
bash setup_h200.sh
source .venv/bin/activate
python train_gpt_oss_120b_qlora.py \
  --train-file data/train_augmented.jsonl \
  --eval-file data/eval.jsonl \
  --test-file data/test.jsonl \
  --output-dir outputs/sql-tuner \
  --budget-hours 4 \
  --max-epochs 3
```

현재 설정: attention q/k/v/o LoRA rank 8, alpha 16, 4bit BnB, BF16, batch 1, accumulation 8, sequence 8192, learning rate 5e-5, cosine, warmup 5%, gradient checkpointing. MoE expert projection은 학습 대상으로 추가하지 않았습니다. 전체 학습 대신 어댑터를 저장합니다.

### 3~5시간에 대한 처리

`--budget-hours`는 3~5 범위의 **학습 실행 예산 상한**입니다. 기본 4시간이며 모델 다운로드/로딩, 사전 baseline 및 최종 test는 포함하지 않습니다. 종료는 optimizer step이 끝날 때 검사하므로 해당 step·평가·체크포인트 저장 시간이 추가될 수 있습니다. 엄밀한 전체 작업 5시간 종료 보장은 없습니다.

첫 epoch의 실제 학습·평가·생성 시간을 측정해 `time_plan.json`에 예상 epoch 수를 계산합니다. 체크포인트와 test에 15%를 남기며 기본 최대 3 epochs를 넘겨 학습하지 않습니다. 초기 컴파일·캐시 때문에 첫 epoch로 추정한 시간이 뒤의 epoch와 다를 수 있습니다. eval loss가 0.001 이상 개선되지 않는 평가가 1회 발생하면 조기 종료합니다.

**증강 데이터도 3~5시간 동안 무조건 반복하는 설정은 제공하지 않았습니다.** 이 규모에서 3시간보다 빨리 종료될 수 있으며, 실제 측정 전에는 3~5시간 분량이라고 보장할 수 없습니다. 시간 계획이 3시간 미만이면 자료 부족을 보고합니다. 더 긴 작업은 독립적인 검증 데이터와 학습 데이터를 추가한 뒤 다시 측정해야 합니다. `--max-epochs`로 상한 변경은 가능하지만 시간을 채우기 위한 다량 반복은 검증 개선과 별개입니다.

## Epoch별 분석

매 epoch마다 eval loss, optimizer-step 로그 train loss 평균, perplexity, 손실 변화, 과적합 신호, 소요 시간, peak GPU allocated memory를 기록합니다. GPU 수치는 예약 메모리나 외부 프로세스 메모리를 포함하지 않습니다.

기본으로 고정 eval 표본 2건을 greedy 생성해 JSON 형식, 필수 필드, 참조 객체 분류 일치, 미검증 DDL/힌트와 after-plan 측정값 생성을 검사합니다. 이는 일부 출력 구조 검사이며 SQL의 의미/성능 검증은 아닙니다. Harmony의 final 채널을 평가하며 analysis만 생성하거나 출력이 잘리면 실패로 기록합니다. 전체 eval 생성 검사를 원하면 `--generation-eval-samples 22`로 지정하세요. 생성 출력이 길면 `--max-new-tokens` 또는 sequence 길이를 조정해야 하며 이 설정은 시간과 메모리에 영향을 줍니다.

- 학습 전: `baseline_metrics.json`
- epoch별 실제 기록: `epoch_analysis.json`, `epoch_analysis.md`
- 표본별 생성 결과: `epoch_*_predictions.jsonl`
- 첫 epoch의 시간 추정: `time_plan.json`
- 실제 token 길이/정답 token 수: `token_statistics.json`
- 실행 버전/GPU 설정: `run_config.json`
- eval loss 기준 최적 어댑터: `best_adapter/`
- 최적 모델 선택 후 1회 test loss: `test_results.json`
- test 전체 생성: `final_test_predictions.jsonl` (생성 검사를 활성화했을 때)
- 실제 학습 시간·선택 checkpoint: `summary.json`

입력/system token의 loss를 마스킹하고 assistant 최종 답변만 학습합니다. 너무 긴 레코드를 자동 절단하지 않고 ID와 길이를 표시해 실패합니다. 완성되지 않은 시간 제한 epoch는 부분 epoch로 기록합니다. test는 checkpoint 선택과 조기 종료에 사용하지 않습니다.

분석 파일을 다시 만들려면:

```bash
python analyze_training.py outputs/sql-tuner
```

손실 감소는 참조 답변 예측 개선입니다. 실제 Oracle에서 원본/후보의 결과 집합 동등성, `ALLSTATS LAST`, buffers, elapsed time, physical reads, DML 부하를 비교해야 SQL 튜닝 성능을 평가할 수 있습니다. 여기서는 Oracle 접속과 DB 실행 평가를 수행하지 않았습니다.

## 재구성과 CPU 검증

```bash
python -m pip install sqlglot==27.14.0
python rebuild_data.py --input-dir sources --output-dir data
python train_gpt_oss_120b_qlora.py --validate-only
python -m unittest discover -p 'test_*.py' -v
```

현재 검증 결과와 원본 코드 문제는 `ANALYSIS.md`에 있습니다. 업로드 문서/코드/데이터 내용은 분석 대상이며 실행 지시로 취급하지 않았습니다.
