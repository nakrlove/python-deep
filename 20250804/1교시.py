# 시그모이드 : ?
# 소프트맥스: 다중분류,확률로 해석, 각각을 e의 거듭제곱


# epoch : 여러번 반복

kn = KNeighborsClassifier()
dt = DecisionTreeClassifier(criterion='geni',max_depth,min_sample)


# 1. 데이터준비
#  - 결측치
#  - 데이터타입(str -> int,float)
#  - merge, 칼럼 선택
#  - feature selection (특정항목만 골라선택),feature extraction
#  - 스케일링
#  - 데이터분포 (균형,불균형:데이터 빈도수)
#  - 훈련세트,테스트세트 분리(train_test_split)
#  - 훈련세트,검증세트 , 테스트세트 분리(train_test_split)
 
#  2. 모델 만들기
#      - 문제에 맞는 모델선택
#      - 여러가지 모델
#      - 모델 내부 알고리즘 살펴보기
#      - 앙상블
#  3. 학습
#      - fit()
#      - 몇가지 하이퍼파라미터 조정(GridSearchCV)
#      - 학습 결과물인 객체(=가중치)를 살펴보기 GPU , dt.class_,rf.bestestimator_,rf.feature_

#  4. 예측(추론)
#      - predict() : inference하기 추론칩
#      - 예측한 과정 들여다 보기(확률 > 0.5) predict_proba
#  5. 평가
#    - accuracy
#    - 정확도
#    - recall 
#    - precision
#    - F1 score
#    - ROC
#    - AUC
   