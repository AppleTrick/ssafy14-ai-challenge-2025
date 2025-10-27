# 🌐 Translated Data

이 디렉토리에는 번역된 데이터 파일들을 넣습니다.

## 📝 파일

### 번역 데이터 파일

다음 파일들을 여기에 추가하세요:

- `train_english.csv` - 영어로 번역된 학습 데이터
- `test_english.csv` - 영어로 번역된 테스트 데이터
- `translation.ipynb` - 번역 노트북 (선택사항)

## 🎯 사용 목적

### 왜 번역을 시도했나?

1. **Qwen 모델 최적화**

   - Qwen 모델은 영어 데이터로 사전 학습됨
   - 영어 데이터에서 더 높은 성능 기대

2. **데이터 다양성**

   - 영어 데이터셋이 훨씬 크고 다양함
   - 모델의 일반화 능력 향상

3. **성능 향상 가능성**
   - 더 나은 이해와 추론 능력
   - 정확도 개선 가능

## 📂 파일 구조

```
data/translated/
├── README.md                # 이 파일
├── train_english.csv        # 번역된 학습 데이터
├── test_english.csv         # 번역된 테스트 데이터
└── translation.ipynb        # 번역 스크립트 (선택)
```

## ⚠️ 중요 사항

**이 실험은 실행하지 못했습니다!**

- 시간 제약으로 실행하지 못한 실험
- 향후 시도할 아이디어
- 실제 프로젝트에서는 사용하지 않음

## 🔗 관련 문서

- [실험적 노트북](../../notebooks/experimental/)
- [결과 및 향후 계획](../../docs/Result.md)

## 💡 번역 예시

### 원본 (한국어)

```
질문: 이 사진 속 운동기구가 설치된 장소는 어디일까요?
선지:
a. 학교 운동장
b. 공원
c. 헬스장 내부
d. 쇼핑몰 내부
정답: b
```

### 번역 (영어)

```
Question: Where is the exercise equipment installed in this photo?
Choices:
a. School playground
b. Park
c. Gym interior
d. Shopping mall interior
Answer: b
```

## 📌 사용 방법 (미실행)

만약 번역 데이터를 사용한다면:

```python
# 번역된 데이터 로드
train_df = pd.read_csv("data/translated/train_english.csv")
test_df = pd.read_csv("data/translated/test_english.csv")

# 일반 데이터와 동일하게 사용
# ...
```
