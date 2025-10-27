# 방법론

## 📋 모델 선택

### 1. 초기 모델: Qwen2.5-VL-3B

**특징**:

- 파라미터 수: 3B
- 아키텍처: Vision-Language Model
- 성능: Base 모델에서 0.23713

**한계**:

- 성능 제한
- 최신 아키텍처 대비 성능 부족

### 2. 최종 모델: Qwen3-VL-4B-Instruct

![Qwen3 Model](Image/qwen3.png)

**선택 이유**:

- 더 나은 아키텍처와 성능
- Instruct 모델로 fine-tuning 적합
- VQA 태스크에 최적화된 구조

**성능 향상**:

- 0.23713 → 0.91615 (약 286% 향상)
- 반 2등 달성

**주요 설정**:

```python
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
IMAGE_SIZE = 224
BATCH_SIZE = 4
GRAD_ACCUM = 4
EPOCHS = 1
LEARNING_RATE = 1e-4
```

## 🔄 학습 파이프라인

### 1. 데이터 전처리

#### Dataset 클래스

```python
class CustomDataset(Dataset):
    def __init__(self, df, processor, image_size=224):
        self.df = df
        self.processor = processor
        self.image_size = image_size

    def __getitem__(self, idx):
        # 이미지 로드
        image = Image.open(self.df.iloc[idx]['path'])
        image = image.resize((self.image_size, self.image_size))

        # 텍스트 처리
        user_text = self.df.iloc[idx]['question']

        return image, user_text
```

#### Collator

```python
class Collator:
    def __init__(self, processor, system_instruction):
        self.processor = processor
        self.system_instruction = system_instruction

    def __call__(self, batch):
        images, texts = zip(*batch)
        # 배치 처리
        inputs = self.processor(
            texts=texts,
            images=images,
            padding=True,
            return_tensors='pt'
        )
        return inputs
```

### 2. 모델 설정

#### LoRA 설정

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
```

#### 양자화 설정

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
```

### 3. 학습 설정

#### 옵티마이저

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)
```

#### Scheduler

```python
from transformers import get_cosine_schedule_with_warmup

num_training_steps = EPOCHS * update_steps_per_epoch
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_training_steps // 10,
    num_training_steps=num_training_steps
)
```

## 🚀 추론 파이프라인

### 1. 배치 추론

```python
# Padding 설정
processor.tokenizer.padding_side = 'left'

# 배치 추론
for inputs in test_loader:
    # 입력 길이 저장
    input_length = inputs['input_ids'].shape[1]

    # 생성
    out_ids = model.generate(
        **inputs,
        max_new_tokens=2,
        pad_token_id=processor.tokenizer.eos_token_id
    )

    # 생성 부분만 추출
    generated_ids = out_ids[:, input_length:]

    # 디코딩
    output_texts = processor.batch_decode(generated_ids)
```

### 2. 답변 추출

```python
def extract_choice(text):
    patterns = [
        r'\b([abcd])\b',
        r'answer[:\s]+([abcd])',
        r'\(([abcd])\)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return 'a'  # 기본값
```

## ⚡ 최적화 전략

### 1. 데이터 로딩 최적화

```python
# DataLoader 설정
train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,              # 병렬 처리
    prefetch_factor=4,          # 미리 준비
    persistent_workers=True,    # 워커 재사용
    pin_memory=True,
    collate_fn=collator
)
```

**효과**:

- GPU 유휴 시간 제거
- 배치 처리 속도 4배 향상

### 2. 이미지 전처리 간소화

```python
# 단일 리사이즈로 통일
image = image.resize((224, 224))
```

**효과**:

- 전처리 시간 50% 단축
- CPU 부담 감소

### 3. 데이터타입 통일

```python
# bfloat16으로 통일
bnb_4bit_compute_dtype = torch.bfloat16
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    # 학습 코드
```

**효과**:

- fp16 ↔ bf16 캐스팅 오버헤드 제거
- 메모리 사용량 감소

### 4. Gradient Checkpointing 비활성화

```python
# 메모리 충분 시 비활성화
# base_model.gradient_checkpointing_enable()  # 주석 처리
```

**효과**:

- 1.5~2배 속도 향상
- 메모리 여유 시 권장

## 🔍 핵심 최적화 코드

### Padding 설정 (중요!)

```python
# Decoder-only 모델은 left padding 필수
processor.tokenizer.padding_side = 'left'
```

**이유**:

- Right padding은 모델이 padding을 정답으로 착각
- Left padding은 모델이 정상 작동

### 입력 제거

```python
# 생성된 토큰만 추출
input_length = inputs['input_ids'].shape[1]
generated_ids = out_ids[:, input_length:]
```

**이유**:

- 배치 디코딩 시 입력까지 포함됨
- 생성 부분만 추출해야 정확한 답변

## 📊 성능 개선 요약

### 학습 최적화

- 배치 처리 시간: 13초 → 3~4초 (4배)
- GPU 활용률: 20% → 80%
- 메모리 사용량: 안정화

### 추론 최적화

- 추론 시간: 최적화 시도
- 배치 크기: 최대한 증가
- Padding 설정: left 적용

### 정확도 향상

![Train Result](Image/train_result.png)

- 초기: 0.23713
- 최고: 0.91615
- 개선율: 286%

## 🎯 모델 비교

| 모델                        | 점수    | 비고       |
| --------------------------- | ------- | ---------- |
| Qwen2.5-VL-3B               | 0.23713 | Base       |
| Qwen3-VL-4B-Instruct        | 0.91615 | Fine-tuned |
| Qwen3-VL-4B-Instruct (병렬) | 0.91358 | 약간 낮음  |
| Qwen3-VL-4B-Instruct (최종) | 0.87294 | 하락       |

## 💡 핵심 교훈

1. **모델 업그레이드**: 최신 아키텍처 사용
2. **Padding 설정**: Decoder-only는 left padding
3. **배치 처리**: 최대한 활용
4. **데이터 로딩**: 병렬 처리 필수

---

**작성일**: 2024년 10월  
**모델**: Qwen3-VL-4B-Instruct  
**최종 점수**: 0.91615 (반 2등) 🏆
