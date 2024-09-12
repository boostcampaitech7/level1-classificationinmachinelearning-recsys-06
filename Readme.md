# 비트코인 시계열 분류 예측

## 1. 프로젝트 구조

```bash
├── Code # AI 모델 학습을 위한 부분
│   ├── config # config.yaml 값 가져 오는 함수 제공
│   ├── dataset # 기존 raw데이터 합치기
│   ├── model # AI 모델 Code ex) LSTM.. 
│   └── pre_process # 모델 학습전 전처리
├── data #.gitignore
│   └── .csv #.gitignore
├── EDA # 개인 EDA 폴더
│   └── {팀원 명} 
│        ├──*.ipynb
├── config-sample.yaml # 하이퍼 파라미터 및 모델&서버 선택을 위한 설정 값
├── .gitignore
├── Readme.md
└── requirements.txt

```

## 2. 라이브러리 버전

**Python 버전**

- 3.12.5

**Library 버전** - (requirements.txt)

```txt
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.4.2
tqdm==4.66.4
xgboost==2.0.3
lightgbm==4.3.0
scipy==1.11.4
plotly==5.22.0
nbformat==5.10.4
```
