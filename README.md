# Title (Please modify the title)
## Team

| ![이정민](https://avatars.githubusercontent.com/u/122961094?v=4) | ![김태현](https://avatars.githubusercontent.com/u/7031901?v=4) | ![문진숙](https://avatars.githubusercontent.com/u/204665219?v=4) | ![강연경](https://avatars.githubusercontent.com/u/5043251?v=4) | ![진 정](https://avatars.githubusercontent.com/u/87558804?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [이정민](https://github.com/lIllIlIIIll)             |            [김태현](https://github.com/huefilm)             |            [문진숙](https://github.com/June3723)             |            [강연경](https://github.com/YeonkyungKang)             |            [진 정](https://github.com/wlswjd)             |
|                            팀장                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |

## 0. Overview
### Environment

- AMD Ryzen Threadripper 3960X 24-Core Processor
- NVIDIA GeForce RTX 3090
- CUDA Version 12.2

### Requirements

- pandas==2.1.4
- numpy==1.23.5
- wandb==0.16.1
- tqdm==4.66.1
- pytorch_lightning==2.1.2
- transformers[torch]==4.35.2
- rouge==1.0.1
- jupyter==1.0.0
- jupyterlab==4.0.9

## 1. Competiton Info

### Overview

- [AIstages - 일상 대화 요약](https://stages.ai/en/competitions/357/overview/description)
- 일상생활에서 대화는 회의, 토의 등에서 다양한 주제와 입장들을 서로 주고받는데, 이를 기억하기에는 한계점이 존재하기에 요약이 필요합니다.
- 일상생활에서 이루어지는 대화를 바탕으로 요약을 효과적으로 생성하는 모델을 개발합니다.

### Timeline

- 7월 25일(금) 10:00 ~ 8월 6일(수) 19:00

## 2. Components

### Directory

- _Insert your directory structure_

e.g.
```
├── code
│   ├── jupyter_notebooks
│   │   └── model_train.ipynb
│   └── train.py
├── docs
│   ├── pdf
│   │   └── (Template) [패스트캠퍼스] Upstage AI Lab 1기_그룹 스터디 .pptx
│   └── paper
└── input
    └── data
        ├── eval
        └── train
```

## 3. Data descrption

### Dataset overview

- 약 12000개의 일상 대화 데이터셋을 이용하여 499개의 대화를 보고 요약
- 최소 2턴부터 최대 60턴까지 대화가 구성되어 있음
- 데이터에는 다양한 형태의 노이즈가 포함 (예를 들어, <br> tag, newline character 오표현 등)

### EDA

- 대화 내 구어체, 불완전 문장, 번역 오류 등의 특이한 언어 패턴 다수 포함
- 대화 내용과 관련없는 tag, 종결사의 반복 등 훈련 시 모델이 헷갈려할 부분이 존재

### Data Processing

- 문장의 시작, 끝을 알려주기 위한 BOS/EOS 토큰 추가
- Max Token 길이 설정 (Encoder : 512, decoder : 100), 다양한 시퀀스 길이를 테스트하였으나 효과는 미미
- 특수 토큰 추가 고려 : # 사이에 있는 단어들을 특수 토큰으로 간주해야할지 → 최종 미사용
- Back Translation 사용 : 한국어 → 영어 (→ 일본어) → 한국어 시도, 하지만 생각보다 좋은 성능을 보여주지는 못함 / 각 과정에서 훈련하는데 시간이 생각보다 오래걸림

## 4. Modeling

### Model descrition

- Kobart, T5, mDialogSum 등의 요약 모델들을 사용
- Kobart의 경우가 가장 좋은 성능을 보여주었음
- T5의 경우 점차 성능이 개선되는 모습을 보여주었지만, 시간이 굉장히 오래걸리고(최소 2~3일 정도로 예상) 훈련이 다 됬다고 해도 Kobart모다 좋을지는 미지수
- mDialogSum (다국어 요약 모델)의 경우 성능이 너무 낮아 첫 훈련 시도 후 폐기
- WnadB로 모델 훈련 관리, WandB Sweep을 통해 어떤 하이퍼 파라미터가 모델 성능에 영향을 주었는지 확인

### Modeling Process

- Hugging Face의 Trainer 클래스로 Fine-Tuning 방식으로 학습 진행
  - 가장 좋았던 설정은 Epochs : 20 / Batch Size : 50 / Learning Late : 1e-5 / Optimizer : AdamW / Scheduler : Cosine
- Prompt Engineering을 이용한 학습 진행
  - 기존 베이스라인의 코드 : 일관된 요약 기준 없이 LLM의 다양한 응답 결과 도출
  - Few-shot 및 One-shot을 통해 대화문에서 어떤 정보를 요약할지 기준을 제시

## 5. Result

### Prompt Engineering VS Fine-Tuning
- 전반적으로 다양한 답변을 제공해주는 Prompt Engineering 방식은 이번 대회의 평가 지표인 Rouge score에 맞춰 특정 방식으로 요약 방식을 맞춰야 한다는 점에서 높은 점수를 기록하기 어려웠음
- 일반적으로 hard voting 방식이 자연어 처리에서는 좋은 성능을 보여주기 어려웠는데, kfold를 이용하여 모델을 훈련하고 최고점 3개를 cosine similarity로 hard-voting한 결과가 가장 좋은 점수를 보여주었음

- Prompt Engineering의 경우

### Leader Board

<img width="966" height="386" alt="image" src="https://github.com/user-attachments/assets/a3fea6ae-30cd-4524-92c6-c000c3975342" />


### Presentation

[AI 부트캠프 13기 NLP 경진대회](https://docs.google.com/presentation/d/1NHAyDUWhEJTWe8n4VmnfUeIyyt6eThcF/edit?slide=id.p8#slide=id.p8)
