# Credit Card Segment Classification

## 1. 📌 프로젝트

신용카드 고객의 금융 활동 데이터를 기반으로 머신러닝 모델을 활용해 고객을 A~E 세그먼트로 분류하는 프로젝트입니다.  불균형 클래스 문제를 고려한 CatBoost 기반 분류 모델을 개발하였으며 최종적으로 macro F1-score 0.63의 성능을 달성하였습니다.

- 대회 링크: [Dacon 신용카드 고객 세그먼트 분류 AI 경진대회](https://dacon.io/competitions/official/236236/overview/description)

## 2. 🧩 기술 스택

- Python 3.10.6  
- pandas, numpy
- scikit-learn
- catboost

## 3. 📁 폴더 및 파일 구조
```
Credit-Card-Segment-Classification/
│
├── dataset/                 # 원본 및 전처리된 데이터 (제공되지 않음)
├── preprocessing.py         # 커스텀 전처리 클래스
├── train.py                 # 모델 학습 및 예측 실행 스크립트
├── utils.py                 # 유틸 함수 모음
├── requirements.txt         
├── README.md                
└── result/                  # 모델 및 파라미터 저장
```
- `dataset/`는 대회 제공 데이터를 저장하는 위치입니다.
- 대회 방침에 의해 데이터는 업로드할 수 없으므로 Dacon에서 직접 데이터를 다운로드 받아 `dataset/` 폴더에 직접 저장해야 합니다. (첨부한 링크로 접속 가능)

## 4. ▶️ 프로젝트 실행 방법

1. **[uv](https://github.com/astral-sh/uv) 설치**
```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
그 이후로 콘솔에 나오는 uv 환경변수 설정 명령어를 복사하고 실행하여 환경변수를 설정합니다.

2. **Python 버전 설정**
```bash
uv venv --python 3.10.6
```
- Python 3.10.6을 이용할 수 있으면 이 과정은 생략할 수 있습니다.

3. **가상환경 실행 및 의존성 설치**
```bash
# On macOS and Linux.
source .venv/bin/activate       

# On Windows. 
.venv\Scripts\activate

uv pip install -r requirements.txt
```

이후 eda 폴더 내의 .ipynb 파일을 모두 실행 후 train.ipynb를 실행하여 데이터 전처리와 모델 학습을 진행할 수 있습니다.

## 5. 📊 주요 성과
- CatBoostClassifier 기반 분류 모델 설계
- Macro F1-score: 0.63
- 클래스 불균형 대응: 클래스 가중치 적용

$$
w_c = \frac{N}{K \cdot n_c}
$$

    - N: 전체 샘플 수  
    - K: 전체 클래스 수  
    - n_{c}: 클래스 c에 속한 샘플 수

- Scikit-learn 기반 커스텀 전처리 파이프라인 구현
