## 문장 내 개체간 관계 추출
문장, 단어에 대한 정보를 통해 ,문장 속에서 단어 사이의 관계를 추론하는 모델을 학습시켜 모델이 단어들의 속성과 관계를 파악하며 개념을 학습하는 것이 목표입니다.

가장 먼저 EDA를 통해 데이터가 가진 특성들을 탐색하였고, Baseline 코드를 기반으로 EDA를 통해 세웠던 가설 및 훈련을 진행하면서 새로 생겼던 가설들을 검증하는 방식으로 학습을 진행하였습니다.

## Enviroment & Requirement

- Enviroment
  - CPU : 13th Gen Intel(R) Core(TM) i7-13700K
  - GPU : NVIDIA GeForce RTX 4070
  - CUDA : 12.6

- Requirement
  - pandas == 2.3.1
  - numpy==2.1.3
  - sklearn==1.7.1
  - torch==2.7.1
  - tqdm==4.67.1
  - transformers==4.54.1

## 구조
```
Project_Root
├── code
│   ├── __pycache__/
│   ├── logs/
│   ├── prediction/
│   ├── results/
│   └── wandb/
│       ├── aug_data.py
│       ├── dict_label_to_num.pkl
│       ├── dict_num_to_label.pkl
│       ├── inference.py
│       ├── load_data.py
│       ├── r_bert_load_data.py
│       ├── r_bert_train.py
│       ├── rbert_inference.py
│       ├── requirements.txt
│       └── train.py
├── dataset
│   ├── test/
│   │   └── test_data.csv
│   └── train/
│       └── train.csv
└── EDA
    └── eda.ipynb
```

## 사용 방법
  1. 터미널에서 프로젝트 폴더 내 code 폴더로 이동
  2. train.py 파일에서 84 번째 line의 MODEL_NAME 지정 / 88번째 lin의 훈련 데이터셋 경로 지정 / 166번째 line의 모델 저장 경로 지정
  3. 'python train.py' 명령어로 훈련 진행 → 지정된 모델 저장 경로에 자동으로 폴더 생성 후 모델 저장
  4. 훈련 완료 후 inference.py 파일에서 68번째 line에서 모델 이름 지정 / 78번째 line에서 테스트 데이터셋 경로 지정 / 91번째 line에서 submission이 저장될 파일 경로 지정 / 98번째 line에서 저장된 모델 경로 지정
  5. 'python inference.py' 명령어로 추론 진행
  6. prediction 폴더 내 submission.csv 파일 저장 완료

> 기본적인 사용 방법은 위와 같고, r_bert_train.py의 경우 train.py와 파일이 비슷한 구조이니 같은 방식으로 진행하시면 됩니다.

## EDA
###
- no_relation이 약 30% 차지
- relation 관계 내 일부 label은 40 ~ 100개 정도의 적은 데이터만을 가짐
- 클래스 불균형이 심함 (relation 관계에서 40 ~ 4300 까지 다양한 label 분포)
 - 클래스 불균형 해결 방안
    - 언더샘플링, 오버샘플링
    - 데이터 증강

- Entity Type 확인
  - Subject Entity : PER / ORG만 나타남
  - Object Entity : PER / ORG 이외에도 다양한 관계 포함 
   
### 문장 길이 분포 분석
![sentence_length](/assets/readme/cartoon.png)
  - 단어 수 기준으로 최소 3단어, 최대 108단어, 평균 20단어의 길이
  - 108 단어 전체를 처리할 수 있도록 여유 토큰 포함 토큰화 시 max_length=128로 설정
  - 전체적으로 데이터들이 비교적 균일하여 훈련에 안정성이 있을 것

### Entity 간 거리 분석
- 평균 거리 : 19자
- 최대 거리 293자 → 최소 300 + position까지 커버 필요
  - BERT 만으로는 해당 position 거리를 커버하기 어려울 수 있음
- Subject와 Object 각각 먼저오는 순서가 거의 비슷한 비율인데, 이는 한국어 특성 상 주어-목적어 순서가 상당히 유연하기 때문
  - Subject와 Object에 Entity Marking을 해줌으로써 순서에 따라 Subject, Object 관계를 나누지 않도록 유도

### 관계-Entity 타입 매핑
![sentence_length](/assets/readme/cartoon.png)
- Subject - Object 조합에서 불가능한 조합이 없음
- ORG / PER로 구성된 관계가 전체의 60%를 차지
  - 조직과 사람 중심의 관계 추출이 중요한 포인트\

> Subject와 Object의 관계 파악 → 세부 관계 예측하는 계층적 모델링 설계 가능

### source 별 특성 차이 분석
![sentence_length](/assets/readme/cartoon.png)
- 모든 source가 비슷한 길이 분포를 가짐
  - 문장 길이로는 소스 구분이 어려움
- Wikipedia
  - no_relation이 압도적 (전체 60% 이상)
  - 실제 관계들이 상대적으로 적음
  - 더 보수적/신중한 라벨링을 했을 것
- WikiTree
  - 실제 관계의 비율이 비교적 높음 (약 50%)
  - no_relation 비율이 상대적으로 낮음
  - 더 적극적인 관계 추출을 했을 것

> 각 source 별로 특화된 관계 패턴이 존재 → 도메인 특화 모델링이 필요
- 생각해 볼 수 있는 모델링 전략
  - 각 source 별 (Policy_briefing 제외) 특화 모델 + 통합모델 앙상블 전략

## 실험 추적 및 평가
wandb를 이용하여 진행하였습니다.

![sentence_length](/assets/readme/cartoon.png)

## 실험 내용
| Strategy | f1_score | auprc|
| --- | --- | --- |
| BaseLine Code | 63.7132 | 57.1291 |
| Entity Marking | 66.1295 | 63.4117 |
| weighted focal | 57.3338 | 59.1966 |
| hierarchical Modeling | 58.3725 | 56.8571 |
| Data augmentation | 61.4066 | 66.2419 |
| Model Change | 63.0017 | 70.1146 |

전체적으로 실험해보았던 기본 조합은 위와 같았습니다. 세부적으로 파악해보면 다음과 같습니다.

먼저, EDA를 기반으로 생각했던 전처리 전략은 아래와 같았습니다.

```
1번 : 클래스 불균형 해결 (가중치로 해소 / 데이터 증강)

2번 : Entity Marking 적용

3번 : source별 학습 전략

4번 : 하이퍼파라미터 조정

5번 : 모델 변경

6번 : 모델링 과정을 먼저 no_relation VS relation 을 구분할 수 있도록 훈련 → relation 내 subject_type
이 PER VS ORG 파악 → 세부 관계 분류
```

해당 전략을 기반으로 아래의 실험들을 진행하였습니다.

***

1. Entity Marking + 하이퍼파라미터 조정

  - micro_f1 약 3점 상승, auprc 약 6점 상승, Entity Markig은 모델이 학습하는 과정에서 도움을 준다는 결과 도출

***

2. 1번 실험 + 클래스 불균형 해소를 위한 weighted_focal 적용

  - no_relation 과 relation의 관계가 불균형하다고 생각 → no_relation과 relation 내 세부 관계들이 불균형의 문제인지를 확인 → 3번 실험

***

3. 2번 실험에서 relation이 있는 관계에만 weighted_focal 적용

- 두 번째 실험과 결과가 거의 같음 → 데이터셋이 불균형하다고 생각하였으나, 모델 입장에서는 개선되지 않았음 → 가중치 조절을 통한 클래스 불균형 해소는 폐기

***

4. 3stage 계층적 모델링

→ 스테이지 분류

  - stage1 : no_relation VS relation 분류
  - stage2 : relation으로 분류된 관계 중 subject type이 PER VS ORG 분류
  - stage3 : relation 관계는 다시 세부 관계 파악

- loss값은 이전 실험들에 비해 줄었으나, f1-score 및 auprc가 더 낮아짐 (성능이 나빠짐)
- 모델이 주요 class인 no_relation에 치우쳐져 학습되었을 가능성이 존재
- 클래스 불균형과 계층적 모델링을 같이 사용하는 방식 실험 필요
- 단, 계층을 3단계가 아닌 2단계로 줄여서 실험(stage1 + stage3)

***

5. 모델 변경

- monologg/koelectra-base-v3-discriminator 모델 사용
	- 실험 결과 성능이 월등히 낮아짐 → Koelectra 모델은 자체 토크나이저를 사용, 해당 토크나이저가 klue 데이터와 맞지 않을 가능성이 존재
	- klue 데이터셋으로 사전훈련된 모델이 더 좋은 성능으로 보일 것으로 예상됨 → 크기가 더 큰 모델인 klue/robert-large로 훈련 시도

- klue/robert-large 모델 사용
	- 배치 크기를 32로 진행하였을 때, 첫 번째 실험과 동일한 결과를 얻었음
	- 배치 크기를 16으로 진행하였을 때 auprc가 크게 증가하였고 (70점), micro_f1은 소폭 감소

6. 데이터 증강을 통한 클래스 불균형 해소

- RMR : Subject, Object 단어 이외의 단어를 MASK 후 BERT 모델이 다시 복원하여 데이터 증강 → auprc는 2점 상승하였으나, micro_f1은 5점 감소함

- RMR + RMI : RMR + 문장의 아무 위치에 MASK 처리후 BERT 모델이 복원 → 위의 표에서 나왔던 점수 갱신 (bert-base 모델)

- RMR + RMI + AEDA : AEDA까지 추가하니 성능이 대폭 감소 →비슷한 구성의 데이터가 많아지면서 모델이 훈련 데이터 및 검증 데이터에 과적합 가능성

7. 데이터 증강 + 2stage 계층적 모델링
- 성능이 대폭 감소 → 계층을 2단계로 줄였으나 여전히 3stage 모델링에서 나타났던 문제들이 그대로 나타남.

8. Entity의 type 정보를 모델의 입력값에 추가 → R-BERT
- 지금까지 토큰화 시 각 entity에 entity임을 표시해주는 마킹만 존재
- 훈련 데이터를 보면 entity의 정보들이 같이 포함되어 있음. → R-BERT
- bert-base로 R-BERT 방식 이용했을 때 기본 BaseLine보다 성능이 안나옴

> klue 기반 모델들이 이번 태스크에서 주어진 데이터셋으로 훈련됨 → 다른 모델들을 이용 시 성능이 잘 나오지 않았음


## 도전 과제

- source 별 학습 전략
  - EDA에서 확인했듯이 source 별 특징이 달랐음
  - 이를 이용하여 앙상블 모델을 훈련했다면 결과가 기대됬을 것

- 계층적 모델링의 성능 부진
  - 모델링 과정을 여러 태스크로 나누어 모델이 예측한다면 더 쉬울것으로 예상하였으나 점수가 잘 나오지 않았음
  - 계층적 모델링을 클래스 가중치와 같이 사용한다면 개선되지 않을까하는 아쉬움 존재
