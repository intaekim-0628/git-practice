# 기존 vLLM 방식으로 SQL Tuner 기동

첨부 스크립트의 네이티브 MXFP4 + BnB 학습 LoRA 구성을 유지했습니다. 병합하지 않습니다. 학습용 venv를 바꾸기보다 기존에 vLLM 기동에 성공한 별도 환경을 사용하세요. torch/vLLM 설치를 이 스크립트에서 변경하지 않습니다.

## 파일

- `serve_vllm_lora.sh`: 기동 및 CLI 호환성 확인
- `verify_adapter_compat.py`: 로컬 베이스/adapter의 설정, 모든 LoRA A/B와 베이스 attention shape, tokenizer token ID 비교
- `test_vllm_config.py`: 모의 헤더를 이용한 검증/명령 구성 테스트

실제 학습 완료 adapter와 네이티브 MXFP4 모델은 저장소에 없습니다. 베이스 경로는 `config.json`, tokenizer.json과 model safetensors가 있는 체크포인트 루트여야 합니다. adapter는 `adapter_config.json`, `adapter_model.safetensors`, `tokenizer.json`을 포함해야 합니다. safetensors는 가중치 전체를 RAM에 올리지 않고 헤더만 읽습니다. 누락·불일치가 있으면 시작하지 않습니다. config를 임의 수정해 검사를 통과시키지 마세요.

## 실행

현재 학습 코드가 저장한 최종 adapter 경로를 사용합니다.

```bash
cd /home/sherpa/source/sql_tuner_package
# 기존에 성공한 vLLM 환경을 먼저 활성화
BASE_MODEL=/home/sherpa/gpt-oss-120b \
LORA_PATH=/home/sherpa/source/sql_tuner_package/outputs/sql-tuner/best_adapter \
PORT=8086 \
bash inference/serve_vllm_lora.sh
```

아직 이전 adapter를 사용하려면 LORA_PATH만 `/home/sherpa/itkim/output-gpt-oss-120b-sql-lora`로 변경합니다. adapter shape/token 검사를 통과해야 합니다. 기존 스크립트의 verify_adapter_compat.py가 첨부되지 않아 새로 작성했으며, 기존 검증기와 동일한 기능이라고 주장하지 않습니다.

GPU 로딩 없이 경로 및 생성 명령을 확인하려면:

```bash
BASE_MODEL=/home/sherpa/gpt-oss-120b \
LORA_PATH=/home/sherpa/source/sql_tuner_package/outputs/sql-tuner/best_adapter \
DRY_RUN=1 bash inference/serve_vllm_lora.sh
```

기본 HOST=0.0.0.0, PORT=8086으로 기존 접근 방식을 유지합니다. 로컬 전용이면 HOST=127.0.0.1을 지정합니다. 외부 접근 시 기존 네트워크 접근 제어를 적용하고 필요하면 `VLLM_API_KEY`를 안전한 환경 설정에 넣습니다. API 키를 출력하지 않으며 인자로 전달하지 않습니다. HTTP 직접 접근이 아니라 운영 환경의 HTTPS 경로를 이용하세요.

## 프로파일과 수정 사항

| PROFILE | max context | 동시 seq | batched tokens | GPU utilization |
|---|---:|---:|---:|---:|
| stable 기본 | 8192 | 2 | 8192 | 0.90 |
| throughput | 16384 | 4 | 8192 | 0.92 |
| long-context | 65536 | 4 | 8192 | 0.90 |

기존 stable의 65536 context 설정은 long-context로 옮겼습니다. 현재 데이터의 최대 길이가 3841 tokens이며 처음에는 8192로 검증하는 편이 적합합니다. long-context 및 throughput의 메모리·속도는 H200에서 실제 확인해야 합니다. MAX_MODEL_LEN, MAX_NUM_SEQS, MAX_BATCHED_TOKENS, GPU_MEMORY_UTILIZATION으로 개별 덮어쓰기가 가능합니다.

기존 하드코딩 `--served-model-name sherpa-yak`는 `BASE_SERVED_MODEL`로 바꿨습니다. 기본 API 이름은:

- 베이스: `gpt-oss-120b`
- adapter: `gpt-oss-120b-sql`

MAX_LORA_RANK는 adapter_config의 r을 읽어 기본 설정합니다. 새 학습 코드의 r=8이면 그대로 8입니다. vLLM 릴리스가 해당 rank/모델 조합을 지원하는지는 실제 시작 로그에서 확인해야 합니다. 설치된 CLI의 전체 help와 사용 플래그를 대조하며 지원되지 않는 플래그를 조용히 버리지 않습니다.

원본의 자동 tool choice는 SQL JSON 응답 테스트에 필요하지 않아 기본 비활성화했습니다. 기존 옵션을 그대로 사용하려면 `ENABLE_TOOLS=1`을 지정하세요.

## 기동 확인과 대표 요청

별도 터미널에서:

```bash
curl --fail http://127.0.0.1:8086/health
curl --fail http://127.0.0.1:8086/v1/models
python inference/compare_models.py \
  --url http://127.0.0.1:8086 \
  --models gpt-oss-120b gpt-oss-120b-sql \
  --data data/eval.jsonl --limit 1 \
  --output outputs/vllm_smoke.jsonl
```

인증을 쓰는 경우 compare_models.py는 `SQL_TUNER_API_KEY` 환경 값을 Authorization 헤더에 사용합니다. 서버의 VLLM_API_KEY와 동일한 값을 보안 환경 설정에 주입하세요. curl에도 Authorization 헤더가 필요합니다. 요청 결과에서 error 없음, finish_reason=stop 및 JSON/객체 분류 결과를 확인합니다. health/models 성공만으로 LoRA의 추론 성공을 판단하지 않습니다.

실제 앱은 같은 endpoint에 `model=gpt-oss-120b-sql`로 호출합니다. 베이스 비교는 `model=gpt-oss-120b`입니다. 앱이 기존 이름 sherpa-yak을 사용하면 BASE_SERVED_MODEL=sherpa-yak을 지정하거나 앱 설정을 수정하세요. 새 adapter 이름까지 베이스 이름으로 요청하면 adapter가 적용되지 않습니다.

## 호환성과 검증 한계

기존 기동 경험을 반영해 네이티브 MXFP4에 adapter를 등록합니다. 구조 검사는 shape와 token ID 차이를 찾지만 **BnB 훈련과 MXFP4 추론의 수치적 동등성이나 튜닝 품질을 입증하지 않습니다.** 동일 eval을 BnB+adapter 서버에서도 생성해 비교하면 양자화 전환 영향을 평가할 수 있습니다. vLLM 내부의 fused QKV 매핑 지원 또한 실제 설치 버전에서 추론으로 확인해야 합니다.

현재 클라우드에서는 shell 구문, dry-run 명령 및 shape/token 오류 검출 테스트를 통과했습니다. 모델/adapter 실물과 GPU가 없어 vLLM 로딩 및 SQL 튜닝 품질은 검증하지 않았습니다. 실제 DB 비교 절차는 DEPLOYMENT_AND_BENCHMARK.md를 참고하세요.
