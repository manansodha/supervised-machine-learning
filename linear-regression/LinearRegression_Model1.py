import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

# Reading the data
data = pd.read_csv('dummy_data.csv')
Y = np.array(data['Close'].values)
data = data.drop("Close", axis="columns")

# Creating the target data
X = np.zeros(data.shape[0])
for f in range(data.shape[0]):
    X[f] = (data["High"][f] + data["Low"][f]) / 2   # Target is mean of highest and lowest price of the day

# Train-test data split
X_train = X[0:round(X.shape[0] * 0.7)]
X_test = X[round(X.shape[0] * 0.7):]
y_train = np.array(Y[0:round(X.shape[0] * 0.7)])
y_test = np.array(Y[round(X.shape[0] * 0.7):])


# Computing cost
def compute_cost(x, y, w, b):
    m = x.shape[0]
    # yhat = np.zeros(m)
    total_cost = 0
    for k in range(m):
        f_wb = w * (x[k]) + b
        # cost = (f_wb - y[k]) ** 2
        total_cost += (f_wb - y[k]) ** 2
    total_cost /= (2 * m)
    return total_cost


# Computing the cost gradient
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * (x[i]) + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


# Computing the descent of the cost gradient
def compute_descent(x, y, w_in, b_in, alpha):
    j = []
    p = []
    w = w_in
    b = b_in
    it = 10000
    for i in range(it):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w -= alpha*dj_dw
        b -= alpha*dj_db
        if i < it:
            j.append(compute_cost(x, y, w, b))
            p.append([w, b])
        if i % math.ceil(it/10) == 0:
            print(f"Iteration {i:4}: Cost {j[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
    return w, b


# Computing the root-mean-square error
def compute_rmse(x_train, y_train_, x_test, y_test_, w_in, b_in, alpha_):
    w, b = compute_descent(x_train, y_train_, w_in, b_in, alpha_)
    m = x_test.shape[0]
    yhat = np.zeros(m)
    for i in range(m):
        yhat[i] = (w*x_test[i] + b)
    error = 0
    for i in range(m):
        error += (yhat[i] - y_test_[i])**2
    mse = math.sqrt(error/m)

    return yhat, mse


# Computing the R2 score
def compute_err(x_train, y_train_, x_test, y_test_, w_in, b_in, alpha_):
    m = x_test.shape[0]
    yhat, mse = compute_rmse(x_train, y_train_, x_test, y_test_, w_in, b_in, alpha_)
    err = 0
    for i in range(m):
        err += (y_test_[i] - np.mean(y_test_))**2
    tss = math.sqrt(err/m)
    r2 = 1 - (mse/tss)
    return r2, mse, yhat


print("")
w1 = 0
b1 = 0
alpha = 1e-5
r2_err, rmse_err, y_pred = compute_err(X_train, y_train, X_test, y_test, w1, b1, alpha)
print("R2 score is:", round(r2_err, 2))
print("RMSE score is:", round(rmse_err, 2))

z = data.tail(y_test.shape[0])
plt.plot(z.index, y_test, label='Original')
plt.plot(z.index, pd.Series(y_pred), c='r', label='Predicted')
plt.legend()
plt.show()


