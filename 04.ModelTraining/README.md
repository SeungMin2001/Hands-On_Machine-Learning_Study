[4장 코드](https://colab.research.google.com/github/rickiepark/handson-ml3/blob/main/04_training_linear_models.ipynb)
# 제 4장 모델훈련
- 모델이 어떻게 작동하는지의 대한 내용을 담은 단원
- 이 단원을 이해하는것이 딥러닝을 이해하고 구축하고 훈련시키는 데 필수임.

### 선형회귀

- MSE(X,hθ) : 데이터X와 가설함수 hθ -> 간편하게 이 책에서는 MSE(θ)라고 표기

##### 정규방정식:
X: 입력데이터 행렬
θ: 가중치 벡터
Xθ: 모델의 회귀방정식
y^: 예측값
y: 실제값
J(θ): 바용함수

행렬 제곱을 하려면 -> x=[1.2.3] 를 제곱하려면 x(전치) * x 이렇게 해주면 제곱효과 나옴.(내적)

