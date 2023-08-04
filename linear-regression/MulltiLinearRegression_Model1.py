import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

# Reading the data
data = pd.read_csv('dummy_data.csv')
prediction = np.zeros(data.shape[0])
new_df = pd.DataFrame(columns=['Closing', 'Prediction'])
new_df['Closing'] = data['Close'].values
data = data.drop("Close", axis="columns")

# Train-test data splitting
X_train = new_df.iloc[0:round(new_df.shape[0] * 0.7), 0]
X_test = new_df.iloc[round(new_df.shape[0]*0.7):, 0]
ytrain = new_df.iloc[5:round(new_df.shape[0] * 0.7) + 5, 0]
ytest = new_df.iloc[round(new_df.shape[0]*0.7)+5:, 0]


# Cost Computing
def compute_cost(x_train, y_train, w, b):
    m = x_train.shape[0]
    x = x_train
    y = np.array(y_train)
    total_cost = 0
    for i in range(m-5):
        z = np.array(x.iloc[i:i+5])
        f_wb = np.dot(w, z) + b
        total_cost += (f_wb - y[i-1])**2
    total_cost /= 2 * (m-4)
    return total_cost


# Computing Cost Gradient
def compute_gradient(x_train, y_train, w, b):
    m = x_train.shape[0]
    n = w.shape[0]
    x = np.array(x_train.iloc[:])
    y = np.array(y_train)
    dj_dw = np.zeros(n)
    dj_db = 0

    for i in range(m-5):
        err = np.dot(x[i:i+5], w) + b - y[i-1]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * x[i+j]

        dj_db += err
    dj_dw = dj_dw / (m-4)
    dj_db = dj_db / (m-4)

    return dj_dw, dj_db


# Computing the gradient descent
def gradient_descent(x_train, y_train, w_in, b_in, alpha_):
    w = w_in
    b = b_in
    j = []
    iterations = 30000
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(x_train, y_train, w, b)

        w = (w - alpha_ * dj_dw)
        b -= alpha_ * dj_db

        if i < iterations:
            z = (compute_cost(x_train, y_train, w, b))
            j.append(z)

        if i % math.ceil(iterations / 10) == 0:
            print(f"Iteration {i:4d}: Cost {j[-1]:8.2f}",
                  f"w1: {w[0]: 0.3e}, w2: {w[1]: 0.3e}, w3: {w[2]: 0.3e}, w4: {w[3]: 0.3e}, w5: {w[4]: 0.3e},",
                  f" b:{b: 0.5e}")

    return w, b


# Computing root-mean-square error
def compute_rmse(x_train, y_train, x_test, y_test, w_in, b_in, alpha_):
    w, b = gradient_descent(x_train, y_train, w_in, b_in, alpha_)
    m = x_test.shape[0]
    yhat = np.zeros(y_test.shape[0])
    for i in range(m-5):
        yhat[i] = np.dot(x_test[i:i+5], w) + b
    error = 0
    for i in range(y_test.shape[0]):
        error += (yhat[i] - y_test.iloc[i])**2
    mse = math.sqrt(error/y_test.shape[0]-4)

    return yhat, mse


# Computing R2 error
def compute_err(x_train, y_train, x_test, y_test, w_in, b_in, alpha_):
    m = x_test.shape[0]
    yhat, mse = compute_rmse(x_train, y_train, x_test, y_test, w_in, b_in, alpha_)
    err = 0
    y_test = np.array(y_test)
    for i in range(len(y_test)):
        err += (y_test[i] - np.mean(y_test))**2
    tss = math.sqrt(err/len(y_test))
    r2 = 1 - (mse/tss)
    return r2, mse, yhat


print("")
w1 = np.zeros(5)
b1 = 0
alpha = 1e-6
r2_err, rmse_err, y_pred = compute_err(X_train, ytrain, X_test, ytest, w1, b1, alpha)

print("R2 score is:", round(r2_err, 2))     # Should be in 0-1 range
print("RMSE score is:", round(rmse_err, 2))     # Should be near 1

# Graphical Representation of the result obtained
plt.plot(y_pred, c='k', label="Predicted")
plt.plot(np.array(ytest), label="Original")
plt.legend(loc='upper left')
plt.show()

