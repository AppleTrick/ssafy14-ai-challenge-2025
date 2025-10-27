# 📓 Notebooks

이 디렉토리는 프로젝트에 사용된 모든 Jupyter 노트북을 포함합니다.

## 📂 파일 설명

### 주요 노트북

#### 1. Baseline.ipynb

**베이스라인 모델**

- 모델: Qwen2.5-VL-3B
- 점수: 0.23713
- 목적: 초기 테스트 및 설정 확인

#### 2. score_0.91615.ipynb ⭐

**최고 성능 달성 모델**

- 모델: Qwen3-VL-4B-Instruct
- 점수: **0.91615** (최고)
- 특징:
  - Qwen3 모델로 업그레이드
  - 최적화된 학습/추론 파이프라인
  - 286% 성능 향상

#### 3. train_0.87294.ipynb

**중간 성능 모델**

- 점수: 0.87294
- 목적: 중간 체크포인트

#### 4. infer_0.87294.ipynb

**추론 코드**

- 최종 제출용 추론 코드

## 📊 성능 비교

| 노트북              | 모델                 | 점수           | 순위 |
| ------------------- | -------------------- | -------------- | ---- |
| Baseline.ipynb      | Qwen2.5-VL-3B        | 0.23713        | 초기 |
| score_0.91615.ipynb | Qwen3-VL-4B-Instruct | **0.91615** ✅ | 최고 |
| train_0.87294.ipynb | Qwen3-VL-4B-Instruct | 0.87294        | 중간 |

## 🚀 사용 방법

### 환경 설정

```bash
# 의존성 설치
pip install -r ../requirements.txt

# Jupyter 실행
jupyter notebook
```

### 노트북 실행

1. 해당 노트북 열기
2. 셀 순서대로 실행
3. 결과 확인

## 📝 노트북 실행 순서

1. **Baseline.ipynb** - 초기 테스트
2. **train_0.87294.ipynb** - 학습
3. **infer_0.87294.ipynb** - 추론
4. **score_0.91615.ipynb** - 최고 성능 달성

## ⚙️ 주요 설정

### 공통 설정

```python
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
IMAGE_SIZE = 384
BATCH_SIZE = 8
GRAD_ACCUM = 2
EPOCHS = 1
LEARNING_RATE = 2e-4
```

### 최적화 옵션

```python
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
```

## 🔧 핵심 코드

### 모델 설정

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# 모델 로드
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager"
)
```

### 학습 설정

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)

model = get_peft_model(model, lora_config)
```

## 📂 디렉토리 구조

```
notebooks/
├── README.md                    # 이 파일
├── Baseline.ipynb              # 베이스라인
├── score_0.91615.ipynb        # 최고 성능
├── train_0.87294.ipynb         # 중간 성능
├── infer_0.87294.ipynb         # 추론
└── experimental/               # 실험적 노트북
    └── README.md
```

## 🔗 관련 문서

- [프로젝트 README](../README.md)
- [방법론](../docs/methods.md)
- [평가](../docs/evaluation.md)

## 💡 팁

- GPU 메모리 부족 시 배치 크기 줄이기
- 학습 시간 단축이 필요하면 에폭 수 줄이기
- 최고 성능을 원하면 score_0.91615.ipynb 사용
