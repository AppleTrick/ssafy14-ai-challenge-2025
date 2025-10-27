# Data Directory

이 디렉토리는 프로젝트에서 사용하는 모든 데이터 파일을 포함합니다.

## 📂 디렉토리 구조

```
data/
├── README.md              # 데이터 디렉토리 소개
└── translated/           # 번역된 데이터
    ├── train_english.csv # 영어로 번역된 학습 데이터
    ├── test_english.csv  # 영어로 번역된 테스트 데이터
    └── translation.py    # 번역 스크립트
```

## 📝 원본 데이터

원본 데이터는 대회 제공 데이터셋입니다:

- `train.csv` - 학습 데이터 (3,887개 샘플)
- `test.csv` - 테스트 데이터
- `train/` - 학습 이미지
- `test/` - 테스트 이미지

## 🌐 번역 데이터

### translated/ 폴더

영어로 번역된 데이터를 포함합니다.

**사용 이유**:

- Qwen 모델이 영어에 더 최적화되어 있음
- 영어 데이터셋이 훨씬 크고 다양함
- 성능 향상 기대

**파일**:

- `train_english.csv`: 영어로 번역된 학습 데이터
- `test_english.csv`: 영어로 번역된 테스트 데이터

**사용 방법**:

```python
# 번역된 데이터 사용
train_df = pd.read_csv("data/translated/train_english.csv")
test_df = pd.read_csv("data/translated/test_english.csv")

# 일반 데이터와 동일하게 사용
# ...
```

## 📊 데이터 통계

### 원본 데이터

- **학습 데이터**: 3,887개
- **테스트 데이터**: 테스트 세트
- **언어**: 한국어

### 번역 데이터

- **학습 데이터**: 3,887개 (영어)
- **테스트 데이터**: 테스트 세트 (영어)
- **언어**: 영어

## ⚠️ 주의사항

1. **번역 데이터는 시도하지 못한 실험**입니다.
2. 실제로는 한국어 원본 데이터만 사용했습니다.
3. 향후 실험을 위한 준비된 데이터입니다.

## 🔗 관련 문서

- [데이터셋 설명](../docs/dataDescription.md)
- [결과 및 향후 계획](../docs/Result.md)
