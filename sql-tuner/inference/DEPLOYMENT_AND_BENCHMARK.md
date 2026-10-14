# H200 모델 실행·병합·SQL 성능 비교

## 권장 경로: 동일 BnB 베이스 + adapter

학습 모델은 `unsloth/gpt-oss-120b-unsloth-bnb-4bit`, LoRA 대상은 q/k/v/o입니다. 먼저 학습 완료 후 `outputs/sql-tuner/best_adapter`를 사용하세요. 중간 checkpoint보다 eval 기준으로 선택한 이 어댑터를 사용합니다. 이 저장소에는 학습 완료 어댑터가 없습니다. 사용자 H200에서 생성한 파일이 필요합니다.

서버는 Unsloth의 adapter 경로 로딩 기능을 사용해 adapter_config에 기록된 베이스를 불러옵니다. `/v1/chat/completions`의 모델 이름 `sql-tuner`는 adapter 활성화, `sql-base`는 **동일하게 로딩된 베이스에서 adapter만 비활성화**합니다. 요청은 직렬화하고 KV 캐시를 재사용하지 않습니다. 이는 양자화/토크나이저 차이 없이 파인튜닝 효과를 비교하기 위한 개발 서버입니다. vLLM의 처리량 최적화 서버와 동일한 성능을 기대하지 마세요.

이미 학습이 성공한 가상환경에서 실행합니다. 새로 Unsloth/torch를 설치하거나 현재 환경을 업그레이드하지 않습니다.

```bash
cd sql-tuner                     # 학습 작업 폴더가 sql_tuner_package이면 그 폴더 사용
source .venv/bin/activate        # 현재 환경에서 실제 쓰는 venv 경로로 조정
python -m pip install -r inference/requirements.txt
python inference/serve_sql_tuner.py --adapter outputs/sql-tuner/best_adapter --port 8000
```

별도 터미널에서 시작 확인과 대표 추론을 수행합니다.

```bash
curl --fail http://127.0.0.1:8000/health
python inference/compare_models.py --data data/eval.jsonl --limit 1 --output outputs/serving_smoke.jsonl
```

`serving_smoke.jsonl`에서 두 모델 모두 `error`가 없고 finish_reason=stop이며 JSON·분류 검사가 통과하는지 확인합니다. health 응답만으로 추론 성공을 판단하지 않습니다. 모델 다운로드/로딩은 처음 시작할 때 필요하며 서버 프로세스는 새 작업에서 다시 시작해야 합니다.

실제 앱에서는 POST `/v1/chat/completions`, model=`sql-tuner`, temperature=0, stream=false를 사용합니다. 입력은 현재 학습 데이터와 같은 system 지시와 JSON user payload(task/input.sql/input.observed_plan/input.metadata)로 구성하세요. 가능하면 실제 plan과 소유자·객체 유형·전체 인덱스 정보를 함께 전달합니다. 서버는 SQL을 데이터베이스에 실행하지 않습니다. 일반 OpenAI 호환 필드 전체나 streaming/tool calling은 지원하지 않습니다.

외부 앱이 다른 서버에 있으면 내부 네트워크 경로를 정하고 `SQL_TUNER_API_KEY`를 환경에 설정한 후 `--host 0.0.0.0`으로 기동합니다. 앱 요청에 Authorization: Bearer 헤더를 전달합니다. 공개 운영 배포에서는 별도의 HTTPS/접근 제어와 동시성·타임아웃 정책을 적용하세요. 비밀 값은 코드/로그/저장소에 넣지 않습니다.

## 병합 가능성

PEFT attention LoRA는 병합 가능한 형태입니다. 최신 Unsloth 소스에는 merged_16bit 및 MXFP4 저장 경로가 있고 최신 vLLM GPT-OSS 클래스에는 SupportsLoRA와 qkv 매핑이 있습니다. 이는 **현재 사용 중인 Unsloth 2026.10.5/vLLM 설치 버전의 모든 조합이 검증됐다는 의미가 아닙니다.** 버전별 GPT-OSS 저장 경로·expert 텐서·양자화 지원을 확인해야 합니다.

120B를 전체 BF16로 풀면 가중치만 약 234GB(218GiB)입니다. KV 캐시/실행 버퍼까지 고려하면 H200 1개에 전체 BF16 모델을 올릴 수 없습니다. 병합 작업의 CPU RAM·임시 공간 요구도 구현에 따라 달라집니다. 따라서 adapter serving을 기본으로 제공하고, BF16 병합은 선택적 export로 분리했습니다.

```bash
python inference/merge_adapter.py --adapter outputs/sql-tuner/best_adapter --output outputs/merged-bf16
```

최소 300GiB 여유 디스크를 검사하지만, 이것이 RAM/전체 임시 파일 용량이나 성공을 보장하지는 않습니다. 인플레이스 `merge_and_unload()`로 BnB 가중치를 덮어쓰는 방식은 사용하지 않습니다. 자동 다운로드/변환에는 검증된 라이브러리의 기본 인증·TLS 경로를 사용합니다. 병합 중 실패하면 adapter serving으로 돌아갈 수 있으며 원본 어댑터는 유지합니다.

H200 1개에서 이 BF16 export를 시험하려면 아래처럼 다시 BnB 4bit로 로딩할 수 있습니다. **재양자화 때문에 adapter 방식과 수치적으로 완전히 같지는 않을 수 있습니다.** 메모리/실제 로딩은 GPU에서 검증해야 합니다.

```bash
python inference/serve_sql_tuner.py --merged-model outputs/merged-bf16 --port 8000
python inference/compare_models.py --models sql-tuner --data data/test.jsonl --output outputs/merged_comparison.jsonl
```

adapter 서버에서 동일 test·temperature·max_tokens로 기록한 결과와 비교합니다. 비교 파일의 `sql-tuner` 출력 및 구조 지표를 ID별로 연결하세요. 먼저 eval로 변환 검증을 하고 test는 최종 확인에 사용하세요. 안정적인 배포가 목적이면 검증되지 않은 병합보다 adapter 경로를 우선 사용합니다.

공식 `openai/gpt-oss-120b`는 MXFP4 expert 형식이며 학습에 쓴 Unsloth BnB 4bit 모델과 같지 않습니다. vLLM에서 공식 MXFP4 모델에 학습 어댑터를 바로 붙이는 경로는 모델 revision, adapter key 매핑과 양자화 차이 검증을 거쳐야 하므로 최초 기본 경로로 제공하지 않았습니다. 이후 사용자가 제공한 기존 vLLM 기동 스크립트를 반영해 별도 serve_vllm_lora.sh를 추가했습니다. VLLM_START.md를 참고하세요. MXFP4 병합/재양자화도 별도 변환 프로젝트로 다뤄야 합니다.

근거로 확인한 upstream 소스(릴리스 지원 증명 대신 구조 확인):
- https://github.com/unslothai/unsloth/blob/main/unsloth/save.py
- https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/gpt_oss.py

## 실제 애플리케이션 SQL 비교

**LLM 응답 개선, SQL 실행 개선, 앱 전체 지연을 별도로 측정하세요.** LLM이 JSON을 잘 만들거나 추론이 빨라졌다고 SQL이 빨라졌다는 뜻은 아닙니다.

1. 실제 앱의 쿼리 10~30개를 선정합니다. 테스트 DB의 실제 스키마/데이터에 맞는 SQL, bind 값·자료형, 쿼리 목적과 반환 컬럼을 고정합니다. 첨부 데이터의 예제 테이블은 실제 앱 스키마와 다를 수 있으므로 그대로 실행하지 않습니다. 현재 test는 2개 SQL 계열뿐이므로 앱 성능 대표성은 부족합니다.
2. 동일 원본 SQL/계획/메타데이터를 base와 adapter 모델에 전달하고 출력 내용을 저장합니다. 원본 SQL 입력에서 사용자/assistant 정답을 제거해야 합니다. `compare_models.py`는 이를 처리하지만 DB 실행은 하지 않습니다.
3. 결과가 `proposed_sql=null`이면 변경 권고 없음으로 기록합니다. 현재 데이터는 부족한 근거에서의 판단 보류 응답이 많으므로 튜닝 SQL이 항상 나온다고 기대하지 않습니다. 근거를 보강하거나 전문가가 후보 SQL을 검토해야 합니다.
4. 후보는 개발/스테이징 Oracle에서 수동 검토 후 검증합니다. generated DDL/힌트/DBMS_STATS를 자동 실행하지 않습니다. SELECT 후보도 결과 동등성을 먼저 검사합니다. NULL, 중복 행, 날짜 경계, 시간대, NLS, 바인드 자료형과 ORDER BY 유무를 포함합니다. 읽기 전용 계정/트랜잭션이 적절하며 Oracle DUAL/읽기 함수가 동등성 문제를 일으킬 수 있습니다.
5. 결과 비교는 같은 데이터 스냅샷에서 전체 fetch를 기준으로 합니다. ORDER BY가 없으면 순서 차이는 허용하되 중복 개수를 보존하는 multiset 비교를 합니다. LOB/부동소수점/시간 자료형은 앱의 정확도 규칙에 따라 비교합니다. 함수 호출이나 실시간 timestamp는 별도 허용 규칙 없이는 동일하다고 판정하지 않습니다.
6. 각 SQL은 warmup 2회 후 원본/후보를 A-B-B-A 순서로 반복합니다. 각 10~30회 이상 실행하고 bind 분포도 대표적으로 선정합니다. 원본과 후보의 세션 설정, schema, optimizer 설정, 데이터·통계 상태 및 동시 부하를 유지합니다. 공유 시스템의 buffer cache를 강제로 flush하지 않습니다. 인덱스 변경과 SQL rewrite 실험은 분리합니다.
7. 실제 cursor의 SQL_ID/child_number를 확보하고 허용된 계측 설정으로 DBMS_XPLAN.DISPLAY_CURSOR의 ALLSTATS LAST를 수집합니다. 예상 EXPLAIN PLAN 비용만 비교하지 않습니다. 서버 elapsed/CPU, buffer gets, physical reads, 실제 rows와 앱 execute/첫 행/전체 fetch 시간을 측정합니다. 계측 오버헤드가 양쪽에 동일하도록 설정하고 누적 통계는 실험 전후 delta를 사용합니다.
8. 변경 후 응답값 동등성 + median/p95 개선 + 다른 bind에서 회귀 없음 + 인덱스 변경의 DML/공간 영향까지 확인한 후보만 앱에 적용합니다. 개선율=(원본-후보)/원본×100이며 SQL 단위별 결과와 전체 요청 분포를 모두 보관합니다.

튜너가 온라인 요청마다 호출되는 앱이면 최종 지연에 LLM 추론 시간을 포함해야 합니다. 일반적으로 SQL 분석은 오프라인/관리자 워크플로로 수행하고, 검증·승인한 SQL을 앱에 배포하는 방식이 적합합니다. 첫 단계는 스테이징 DB와 앱의 기존 실행 경로를 이용해 아래 case 목록을 채우는 것입니다.

`sql_benchmark_cases.template.json`에 사례 기록 구조를 제공합니다. 실제 DB 연결과 측정은 앱의 데이터 접근 코드에 붙여야 하며 이 세션에서는 Oracle/GPU 실행을 수행하지 않았습니다.
