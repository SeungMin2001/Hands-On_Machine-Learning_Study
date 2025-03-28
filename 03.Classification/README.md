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

- sklearn의 많은 알고리즘들은 numpy로 주는게 더 좋음.
- 데이터를 자세히 조사하기 전에 항상 테스트 세트를 만들고 따로 떼어놓아야 한다! mnist 데이터셋은 이미 나눠져 있는 세트 60000기준으로
```py
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
```
<br>

### 이진 분류기 훈련
```py
y_train_5 = (y_train == '5')  # 5는 True고, 다른 숫자는 모두 False
y_test_5 = (y_test == '5')

# y_train,y_test 리스트는 각각의 데이터에 대한 정답을 갖고있다. 예를 들어 [1,4,2,5,2,3,...] 이렇게
# 그중 값이 ==5 이면 true반환 즉 y_train_5,y_test_5 는 true,false로 이루어진 불리언 리스트가 된다. [True,False,True,True...] 이런식으로
```
<br>

- 확률적 경사 하강법
- 기울기를 활용한 손실함수 수렴하게하는 방법. 난수를 고정시킴으로써 고정된 데이터셋에서 손실함수를 0으로 수렴시킴
- 데이터가 많을수록 GD->SGD 를 채택해서 사용
```py
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
```
<br>

```py
# X_train[0] 은 단일원소 리스트가 아니기때문에 []를 감싸줘서 보내야함
sgd_clf.predict([X_train[0]])
```
<br>

### 성능측정
- 교차검증을 사용한 정확도 측정 - k-폴드 교차 검증
```py
# 교차검증 직접 구현

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3)  # 데이터셋이 미리 섞여 있지 않다면
                                       # shuffle=True를 추가하세요.
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))
```
<br>

- 
```py
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy") #cv=교차검증을 위한 폴드 수
```
<br>

- 더미분류기 사용 - 더미는 학습모델이 아닌 주로 다른 모델과 비교할때 기준선 같은 역할을 한다.
```py
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier()
dummy_clf.fit(X_train, y_train_5)
print(any(dummy_clf.predict(X_train))) #이 값이 False가 나왔는데 이 뜻은 전부 False라는 뜻이다.(any이므로)

cross_val_score(dummy_clf, X_train, y_train_5, cv=3, scoring="accuracy")

# DummyClassifier는 기본적으로 most_prequent이므로 더 많이 나온 답으로 전부를 예측때려버린다.
# 즉 True 보다 False값이 더 많았을것이고 모든 경우의 답을 False로 예측했을 것이다.
# 더미결과 정확도가 90% 이상으로 나오는데 이 이유는 데이터셋에 숫자 5가 10%정도 있기때문에 False값에 대한 정답이 90%가 나오는 것이다.
```
<br>

- 불균형한 데이터를 다룰때에는 정확도보단 오차행렬로 평가하는게 더 좋다.
- 오차행렬을 만들려면 실제 타깃과 비교할수 있도록 예측값을 만들어야 한다.
- 테스트 세트로 예측을 만들수 있지만 여기서 사용하면 안됨. 테스트 세트는 프로젝트 맨 마지막에 분류기가 출시 준비를 마치고 나서 사용해야하기 때문.

- cross_val_predict 를 사용해서 각 테스트 폴드에서 얻은 예측을 반환, 예측은 인자값으로 넣어준 모델을 기반으로 나온 값임
```py
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3) # sgd_clf모델사용, 훈련세트개수 3개
# 교차검증에서 폴드개념은 축구 리그 구조와 같음
```
<br>

- confusion_matrix() 즉 오차행렬을 구하려면 예측값이 필요!, 단 테스트세트로 만들면 안됨!
- 따라서 cross_val_predict()을 통해 k-폴드 교차검증을 사용해서 예측값들을 만들어내고 그 값을 confusion_matrix()에 넣어줄거임
- 그래서 위 코드에서 y_train_pred 변수에 새로 생성된 깨끗한 예측값을 담아준거임, 이걸 confusion_matrix에 넣어주면 됨
- 오차행렬에 기준값은 데이터셋에 target값에 데이터에 따라 바뀌는것을 알수 있다.
- 지금처럼 target데이터가 True,False처럼 0과1로 이루어져있기 때문에 오차행렬도 0과1로 구분하여 카운팅 하는것이다.
```py
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_train_5, y_train_pred)
```
<br>

```py
y_train_perfect_predictions = y_train_5  # 완벽한 분류기일 경우
confusion_matrix(y_train_5, y_train_perfect_predictions)
```
<br>

- 오차행렬에 정밀도와 재현율
- 정밀도: 모델이 '5'라고 예측한 것중에서 실제로 '5'가 맞았는지의 비율
- 재현율: 모델이 실제 '5'인 데이터 중에서 얼마나 많은 '5'를 올바르게 예측했는지의 비율
```py
from sklearn.metrics import precision_score, recall_score

precision_score(y_train_5, y_train_pred)  # == 3530 / (687 + 3530)
# 추가 코드 – TP / (FP + TP) 식으로 정밀도를 계산합니다
cm[1, 1] / (cm[0, 1] + cm[1, 1])

recall_score(y_train_5, y_train_pred)  # == 3530 / (1891 + 3530)
# 추가 코드 – TP / (FN + TP) 식으로 정밀도를 계산합니다
cm[1, 1] / (cm[1, 0] + cm[1, 1])
```
<br>

- F1점수(F1 Score) : 정밀도와 재현율의 조화평균
- 조화평균이란 낮은 값에 훨씬 더 높은 비중을 둬서 평균을 구하는 방식이다. f1_score()를 호출하면 구할수 있다.
- 즉 오차행렬을 구한뒤 F1점수, 조화평균을 구해서 성능 측정을 할수도 있다는 뜻이다.
```py
from sklearn.metrics import f1_score

f1_score(y_train_5, y_train_pred)
```
<br>

- 정밀도/재현율 트레이드오프
- 정밀도와 재현율을 모두 얻을수는 없다. 정밀도를 올리면 재현율이 줄고, 그 반대도 마찬가지이기 때문 이를 정밀도/재현율 트레이드오프라 말한다.

- decision_function()으로 원하는 임곗값을 정해 예측을 만들수 있다하는데 아직 이해가 잘 되지 않는다.
- 책을 한바퀴 돌리고 다시와서 공부하자.

- 임곗값을 높이면 재현율이 줄어들고 정밀도가 올라간다. 반대로 낯추면 재현율이 높아지고 정밀도가 낮아진다.

- 적절한 임곗값을 구하기 위해서는 먼저 cross_val_predict() 을 사용해 모든 훈련세트에 있는 모든 샘플의 점수를 구해야한다.
```py
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")
```
<br>

- 그 다음 precision_recall_curve()를 사용해 모든 임곗값에 대한 정밀도와 재현율을 계산할수 있다.
```py
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
```
<br>

- 이제 matplotlib을 통해 임곗값의 함수로 정밀도와 재현율을 그릴수 있다.
```py
plt.figure(figsize=(8, 4))  # 추가 코드
plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
plt.vlines(threshold, 0, 1.0, "k", "dotted", label="threshold")

# 추가 코드 – 그림 3–5를 그리고 저장합니다
idx = (thresholds >= threshold).argmax()  # 첫 번째 index ≥ threshold
plt.plot(thresholds[idx], precisions[idx], "bo")
plt.plot(thresholds[idx], recalls[idx], "go")
plt.axis([-50000, 50000, 0, 1])
plt.grid()
plt.xlabel("Threshold")
plt.legend(loc="center right")
save_fig("precision_recall_vs_threshold_plot")

plt.show()
```
<br>

- 좋은 정밀도/재현율 트레이도오프를 선택하는 방법은 재현율에 대한 정밀도 곡선을 그리는거임.
```py
import matplotlib.patches as patches  # 추가 코드 – 구부러진 화살표를 그리기 위해서

plt.figure(figsize=(6, 5))  # 추가 코드

plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")

# extra code – just beautifies and saves Figure 3–6
plt.plot([recalls[idx], recalls[idx]], [0., precisions[idx]], "k:")
plt.plot([0.0, recalls[idx]], [precisions[idx], precisions[idx]], "k:")
plt.plot([recalls[idx]], [precisions[idx]], "ko",
         label="Point at threshold 3,000")
plt.gca().add_patch(patches.FancyArrowPatch(
    (0.79, 0.60), (0.61, 0.78),
    connectionstyle="arc3,rad=.2",
    arrowstyle="Simple, tail_width=1.5, head_width=8, head_length=10",
    color="#444444"))
plt.text(0.56, 0.62, "Higher\nthreshold", color="#333333")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.axis([0, 1, 0, 1])
plt.grid()
plt.legend(loc="lower left")
save_fig("precision_vs_recall_plot")

plt.show()

# 그래프를 보면 재현율 80% 근처에서 정밀도가 급격하게 줄얻르기 시작함.
# 이 하강점 직전을 정밀도/재현율 트레이드오프로 선택하는게 좋음
```
<br>

- 정밀도 90% 달성하는것이 목표라고 했을때 그래프에서 찾는건 한계가 있음.
- 그래서 넘파이배열의 argmax()를 사용하여 구할수 있음
```py
# 이 메서드는 최댓값의 첫번째 인덱스를 반환함
idx_for_90_precision = (precisions >= 0.90).argmax()
threshold_for_90_precision = thresholds[idx_for_90_precision]
threshold_for_90_precision
```
<br>

- (훈련세트에 대한) 예측을 만들려면 분류기의 predict() 대신 이 코드를 실행해야함. 새로운 임곗값을 기준으로 예측을 만드는 과정임.
```py
y_train_pred_90 = (y_scores >= threshold_for_90_precision)

precision_score(y_train_5, y_train_pred_90)

recall_at_90_precision = recall_score(y_train_5, y_train_pred_90) # 정밀도 90%를 달성한 분류기를 만든거임.
# 누군가 99% 정밀도를 달성하자 라고 말하면 반드시 재현율 얼마에서? 라고 물어봐야 한다.
```
<br>

- ROC 곡선
