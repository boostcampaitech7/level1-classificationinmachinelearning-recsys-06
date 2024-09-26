# 비트코인 상승/하락 시계열 분류 예측 프로젝트

<details>
    <summary> 프로젝트 구조</summary>

```bash
├── Code # AI 모델 학습을 위한 부분
│   ├── config # config.yaml 값 가져 오는 함수 제공
│   ├── dataset # 기존 raw 데이터 합치기
│   ├── model # AI 모델 Code ex) Light GBM, XGBoost
│   └── pre_process # 모델 학습전 전처리
├── data #.gitignore
│   └── .csv #.gitignore
├── EDA # 개인 EDA 폴더
│   └── {팀원 명} 
│        ├──*.ipynb
├── config-sample.yaml # 하이퍼 파라미터 및 모델 & 서버 선택을 위한 설정 값
├── .gitignore
├── Readme.md
└── requirements.txt
```

</details>

<details>
    <summary> 라이브러리 버전</summary>

**Python 버전 : 3.12.5**

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

</details> 


## 목차
1. 프로젝트 소개
2. 진행 및 협업 방식
3. EDA 및 피처엔지니어링
4. 사용한 모델
5. 최종 선택
6. 추가 정보

##  1. 프로젝트 소개
* 프로젝트 기간 : 2024/09/10 ~ 2024/09/26

* Boostcamp RecSys 6조 팀원 : [김건율](https://github.com/ChoonB), [백우성](https://github.com/13aek), [유대선](https://github.com/xenx96), [이제준](https://github.com/passi3), [황태결](https://github.com/minari-c)

* 데이터 정보 : [Cryptoquant Catalog](https://cryptoquant.com/ko/catalog) 참고

* 프로젝트 목표
> [upstage](https://stages.ai/)의 [비트코인 상승/하락 시계열 분류 예측 대회](https://stages.ai/competitions/313/overview/description) 참가를 위한 프로젝트. <br>
비트코인의 시간별 Market Data와 Network Data를 이용해 다음 시점의 등하락을 분류한다. <br>
2023년의 시간 별 데이터들을 학습해 2024년 1~4월 동안의 다음 시간 비트코인 등락률을 아래의 클래스에 맞게 분류한다.

<details>
    <summary> 타겟 값 </summary>

| 클래스 | 설명           | 등락률         |
|--------|----------------|--------------|
| 0      | 하락           | -0.5% 미만   |
| 1      | 소폭 하락      | -0.5% ~ 0%   |
| 2      | 소폭 상승      | 0% ~ 0.5%    |
| 3      | 상승           | 0.5% 이상    |

</details>


## 2. 진행 및 협업 방식
### 프로젝트 진행
1. 비트코인 도메인 지식을 위한 스터디 진행 후, 주어진 데이터의 각 column에 대해서 토론.
2. 프로젝트를 위한 기본 구조 설립 코드 작성. (유대선)
3. 팀원 별로 EDA와 feature engineering을 진행.
4. 각자 도출한 결과에 대해 공유하고 토론을 진행해 feature와 model 선택
5. 모델 훈련 및 하이퍼파라미터 튜닝
6. 최종 제출 선택

### 협업 방식
* Slack : 팀 간 실시간 커뮤니케이션, 이슈 공유, 질의 응답을 위한 소통 채널
* Zoom : 정기적인 회의와 토론을 위해 사용
* GitHub : 버전 관리와 코드 협업을 위해 사용. 각 팀원은 EDA를 제외하면 기능 단위로 이슈와 브랜치를 만들어 작업했고, Pull Request를 통해 코드 리뷰 후 병합하는 방식으로 진행

## 3. EDA 및 피처 엔지니어링
[EDA 및 피처 엔지니어링 정리 링크](https://miniature-smelt-728.notion.site/EDA-Feature-Engineering-10d68372ae8c80308138c03e13766f43)

## 4. 사용한 모델
1. XGBoost
* 모델 선택 이유 : 예측 정확도가 높고, 결측치를 자동으로 처리를 할 수 있고, 복잡한 비선형 관계를 잘 포착해 금융 시계열 데이터에 효과적이라 선택.
2. LightGBM
* 모델 선택 이유 : 빠른 학습 속도와 메모리 효율성으로 대규모 데이터 셋에 적합하고, 시계열 데이터에 강점이 있어 선택.
3. LSTM(회귀)
* 모델 선택 이유 : 회귀모델로 종가를 예측해서 최종 제출시에 해당하는 종가 등락률을 클래스로 변경. 시계열 데이터의 장기 패턴을 효과적으로 포착하여 미래 가격 변동 예측에 유용하다고 판단해 선택.
* 사용 결과 : 훈련 시간이 길고, validation 결과 종가가 -log 함수처럼 값이 하락 수렴하는 추세가 나와 적합하지 않다고 판단.

## 5. 최종 선택
1. XGBoost
<details>
    <summary> 하이퍼파라미터 세팅 </summary>

```json
{
    "objective": "multi:softprob",  # 다중 클래스 분류
    "num_class": len(y_train_resampled.unique()),  # 클래스 개수
    "eval_metric": "mlogloss",  # 손실 함수 (멀티클래스 로지스틱 손실)
    "max_depth": 7,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42
}
```
</details>
2. LightGBM

<details>
    <summary> 하이퍼파라미터 세팅 </summary>

```json
{
    "boosting_type": "gbdt",
    "objective": "multiclass",
    "metric": "multi_logloss",
    "num_class": 4,
    "num_leaves": 50,
    "learning_rate": 0.05,
    "n_estimators": 30,
    "random_state": 42,
    "verbose": -1,
}
```
</details>
