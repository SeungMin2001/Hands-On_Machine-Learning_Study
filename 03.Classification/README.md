[3장 코드](https://colab.research.google.com/github/rickiepark/handson-ml3/blob/main/03_classification.ipynb)
# 제 3장 분류

### MNIST - (70,000개의 작은 숫자 이미지 데이터셋)
```py
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', as_frame=False)

def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")

some_digit = X[0]
plot_digit(some_digit)
plt.show()

# fetch_openml() 는 입력을 판다스 데이터프레임, 레이블을 판다스 시리즈로 반환.
# 이 말은 mnist.data->dataframe , mnist_target->series 로 바꾼다는 소리.
# 하지만 mnist는 이미지이므로 dataframe->넘파이 배열로 바꿔주는개 편함
# 그래서 as_frame=False를 넣어줘서 넘파이배열로 가져옴
# 이렇게 가져오면 1차원 넘파이 배열이기 땨문에 이미지데이터로 바꾸려면 2차원이여야함.
# 그래서 reshape로 픽셀수의 맞게 28*28로 reshape를 해주고 imshow()로 이미지 데이터로 바꿔주는 코드
```
<br>



