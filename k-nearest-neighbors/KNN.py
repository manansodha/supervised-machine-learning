import random
import numpy as np
import pandas as pd

# Reading the data
df = pd.read_csv("iris.data", header=None)
df.columns = ["sepal length", "sepal width", "petal length", "petal width", "class"]

n = df.shape[0]
n_test = int(n * 0.2)

# Changing named class values to numerical values
iris_class = np.zeros(n, dtype="int32")
for i in range(n):
    if df["class"][i] == "Iris-setosa":
        iris_class[i] = -1
    if df["class"][i] == "Iris-versicolor":
        iris_class[i] = 0
    if df["class"][i] == "Iris-virginica":
        iris_class[i] = 1
df["iris class"] = iris_class

# Random train-test split in 80/20 ratio
random_index = []
for i in range(n_test):
    x = random.randint(0, 149)
    while x in random_index:
        x = random.randint(0, 149)
    random_index.append(x)

random_index.sort()
test_set = pd.DataFrame(index=["sepal length", "sepal width", "petal length", "petal width", "class", "iris class"])
for i in range(n_test):
    test_set[i] = df.iloc[random_index[i], 0:]
y = test_set.transpose()
y = y.drop(columns='class')
X = df.drop(index=random_index)
X = X.drop(columns='class')


# Calculating the mode
def max_freq(lst):
    minus = 0
    zero = 0
    plus = 0
    for j in range(lst.shape[0]):
        if lst.iloc[j, 1] == -1:
            minus += 1
        elif lst.iloc[j, 1] == 0:
            zero += 1
        else:
            plus += 1
    z = max(minus, zero, plus)
    if z == minus:
        return -1
    elif z == zero:
        return 0
    else:
        return 1


# KNN model
def knn(train, test):
    k = 15  # Optimal Value, Accuracy changes with change in k
    a = train.shape[0]
    b = test.shape[0]
    l = np.array(train.iloc[:, 0], dtype='float64')
    w = np.array(train.iloc[:, 1], dtype='float64')
    lt = np.array(test.iloc[:, 0], dtype='float64')
    wt = np.array(test.iloc[:, 1], dtype='float64')
    oc = []
    for f in range(b):
        di = []
        cla = []
        distance = pd.DataFrame(index=range(120), columns=['eu', 'class'])
        for j in range(a):
            xx = round((l[j] - lt[f]) ** 2, 2)
            yy = round((w[j] - wt[f]) ** 2, 2)
            eu = (xx + yy) ** 0.5
            di.append(eu)
            cla.append(train.iloc[j, train.shape[1] - 1])
        distance['eu'] = di
        distance['class'] = cla
        z = distance.sort_values(by=['eu'])
        k_top = z.head(k)
        occ = max_freq(k_top)
        oc.append(occ)

    return oc


t = knn(X, y)
y['Prediction'] = t     # Adding the list of predicted values to target dataframe


# Computing the confusion matrix
def confusion(pred):
    q = pred.sort_values(by=['iris class', 'Prediction'])

    name = ['Setosa', 'Versicolor', 'Virginica']
    cm = pd.DataFrame(index=name, columns=name)

    setosa = [0, 0, 0]
    virginica = [0, 0, 0]
    versicolor = [0, 0, 0]
    m = q.shape[0]
    z = q.shape[1]
    for j in range(m):
        if q.iloc[j, z - 1] == -1:
            if q.iloc[j, z - 2] == -1:
                setosa[0] += 1
            elif q.iloc[j, z - 2] == 0:
                setosa[1] += 1
            else:
                setosa[2] += 1
        elif q.iloc[j, z - 1] == 0:
            if q.iloc[j, z - 2] == -1:
                versicolor[0] += 1
            elif q.iloc[j, z - 2] == 0:
                versicolor[1] += 1
            else:
                versicolor[2] += 1
        else:
            if q.iloc[j, z - 2] == -1:
                virginica[0] += 1
            elif q.iloc[j, z - 2] == 0:
                virginica[1] += 1
            else:
                virginica[2] += 1
    cm['Setosa'] = setosa
    cm['Versicolor'] = versicolor
    cm['Virginica'] = virginica

    return cm


print(y)
confusionMatrix = confusion(y)
print(confusionMatrix, '\n')

tp = confusionMatrix.iloc[0, 0]     # True Positive
fn = confusionMatrix.iloc[0, 1] + confusionMatrix.iloc[0, 2]    # False Negative
fp = confusionMatrix.iloc[1, 0] + confusionMatrix.iloc[2, 0]    # False Positive
# True Negative
tn = confusionMatrix.iloc[1, 1] + confusionMatrix.iloc[1, 2] + confusionMatrix.iloc[2, 1] + confusionMatrix.iloc[2, 2]

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = (2 * precision * recall) / (precision + recall)
accuracy = (tp + fn) / (tp + fp + tn + fn)
specificity = tn / (tn + fp)

print(precision, recall, f1, accuracy, specificity)
