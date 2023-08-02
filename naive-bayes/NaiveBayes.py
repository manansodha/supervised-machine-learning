import pandas as pd
import numpy as np

# Reading and filtering the data
data = pd.read_csv("diabetes.csv")
s = data.shape[0]
q = data.shape[1]
t = []
for i in range(s):
    if data.iloc[i, q-1] == 1:
        t.append("Yes")
    else:
        t.append("No")
data = data.drop(columns='diabetes')
data['diabetes'] = t

# Train-Test data split
data_train = data.iloc[:round(s*0.7), :]
data_test = data.iloc[round(s*0.7):, :]


# Computing the predictor table
def predictor(x_train):
    m = x_train.shape[1]

    yn = x_train.iloc[:, m-1].value_counts()
    yes = yn[0]
    no = yn[1]
    p_yes = yes / (yes + no)    # probability of yes
    p_no = no / (yes + no)      # probability of no

    titles = []
    final = pd.DataFrame(index=['Yes', 'No'])
    # calculating values of labels
    for i in range(m-1):
        a = pd.crosstab(x_train.iloc[:, i], x_train.iloc[:, m-1])
        titles.append(a)

    # calculating yes and no probability of each label p(label|yes) p(label|no)
    for title in titles:
        for i in range(title.shape[0]):
            ne = title.iloc[i, 0] / no
            ye = title.iloc[i, 1] / yes
            final[title.index[i]] = [ye, ne]

    z = final.transpose()

    return z, p_yes, p_no


# Naive Bayes Algorithm implementation
def nb(x_train, x_test):
    m = x_train.shape[0]
    n = x_test.shape[0]

    predict, yes, no = predictor(x_train)

    y = x_test.iloc[:, x_test.shape[1]-1]
    x_test = x_test.iloc[:, :x_test.shape[1]-1]

    predict_df = pd.DataFrame(columns=['Original', 'Predicted'])
    predict_df['Original'] = y
    name = predict_df.iloc[:, 0].unique()
    lst = []
    for i in range(n):
        conditions = x_test.iloc[i, :]
        pyes = yes
        pno = no
        for j in range(conditions.shape[0]):
            pyes *= predict.loc[conditions.iloc[j], name[1]]    # p(yes|label)
            pno *= predict.loc[conditions.iloc[j], name[0]]     # p(no|label)

        # Normalising the probability values
        norm_p = pyes / (pyes + pno)
        norm_n = pno / (pyes + pno)
        if norm_p > norm_n:
            lst.append('Yes')
        else:
            lst.append('No')

    predict_df['Predicted'] = lst

    return predict_df


# Computing Confusion Matrix
def confusion(x_train, x_test):
    # columns are the actual value and rows are the predicted value
    z = nb(x_train, x_test)
    m = z.shape[0]
    confusion_df = pd.DataFrame(index=["Yes", "No"], columns=["Yes", "No"])
    yes = np.zeros(2)
    no = np.zeros(2)
    for i in range(m):
        o = z.iloc[i, 0]
        p = z.iloc[i, 1]
        if o == p:
            if o == 'Yes':
                yes[0] += 1
            else:
                no[1] += 1
        else:
            if o == 'Yes':
                yes[0] += 1
            else:
                no[0] += 1

    confusion_df['No'] = no
    confusion_df['Yes'] = yes

    return confusion_df


confusionMatrix = confusion(data_train, data_test)

tp = confusionMatrix.iloc[0, 0]     # True Positive
fp = confusionMatrix.iloc[0, 1]     # False Negative
fn = confusionMatrix.iloc[1, 0]     # False Positive
tn = confusionMatrix.iloc[1, 1]     # True Negative

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = (2 * precision * recall) / (precision + recall)
accuracy = (tp + fn) / (tp + fp + tn + fn)
specificity = tn / (tn + fp)

print(confusionMatrix)
print("\nPrecision:", precision,
      "\nRecall", recall,
      "\nF1:", f1,
      "\nSpecificity", specificity)
# columns are the actual value and rows are the predicted value
