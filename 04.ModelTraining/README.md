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

행렬 제곱을 하려면 -> x=[1.2.3] 를 제곱하려면 x(전치) * x 이렇게 해주면 제곱효과 나옴.(내적)<br>

<img src="https://github.com/SeungMin2001/Hands-On_Machine-Learning_Study/blob/main/images/IMG_0252.jpg" width="300" height="400">
<br><br>

```py
# 정규방정식
#------------------------------------------------- 랜덤 선형데이터 생성
import numpy as np

np.random.seed(42)  # 코드 예제를 재현 가능하게 만들기 위해
m = 100  # 샘플 개수
X = 2 * np.random.rand(m, 1)  # 열 벡터
y = 4 + 3 * X + np.random.randn(m, 1)  # 열 벡터
#------------------------------------------------- 화면에 띄우기
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.plot(X, y, "b.")
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.axis([0, 2, 0, 15])
plt.grid()
save_fig("generated_data_plot")
plt.show()
#------------------------------------------------- 정규방정식 적용.
from sklearn.preprocessing import add_dummy_feature

X_b = add_dummy_feature(X)  # 각 샘플에 x0 = 1을 추가합니다.
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
# 여기서 @이란 행렬곱셈 연산자임.
# .T는 전치행렬을 취해주는 메서드임.
#------------------------------------------------- 
```
