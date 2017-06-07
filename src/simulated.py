from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set()

# load the skearn iris dataset
def simulated_data():
    iris = datasets.load_iris()
    class_1 = iris.data[iris.target == 0]
    class_2 = iris.data[iris.target == 1]
    label_1 = iris.target[iris.target == 0]
    label_2 = iris.target[iris.target == 1]

    # convert 0 to -1
    label_1 = label_1 - 1
    X = np.concatenate((class_1, class_2), axis=0)
    y = np.concatenate((label_1, label_2), axis=0)

    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


# compute the gradient for the logistic regression model
def compute_grad(beta, lambduh, X, y):
    n = X.shape[0]
    yX = y[:, np.newaxis] * X
    denom = 1 + np.exp(yX.dot(beta))
    grad_beta = (-1 / n) * np.sum(yX / denom[:, np.newaxis], axis=0) + 2 * lambduh * beta
    return grad_beta


# compute the objective function for the logistic regression model
def compute_obj(beta, lambduh, X, y):
    n = X.shape[0]
    obj = (1 / n) * np.sum(np.log(np.exp(-y * X.dot(beta)) + 1)) + lambduh * np.linalg.norm(beta) ** 2
    return obj


# back tracking algorithm
def back_tracking(beta, lambduh, t, X, y, alpha=0.5, gamma=0.8, max_iter=100):
    grad_beta = compute_grad(beta, lambduh, X, y)
    norm_grad_beta = np.linalg.norm(grad_beta)
    found_t = 0
    iteration = 0
    while (found_t == 0 and iteration < max_iter):
        if compute_obj(beta - t * grad_beta, lambduh, X, y) < \
                        compute_obj(beta, lambduh, X, y) - alpha * t * norm_grad_beta ** 2:
            found_t = 1
        elif (iteration == max_iter):
            print("Reach the maximum number of iterations.")
            break
        else:
            t = t * gamma
            iteration = iteration + 1
    return t


# gradient descent algorithm
def gradient(beta, t_init, lambduh, X, y, max_iter=1000):
    beta_vals = beta
    grad_beta = compute_grad(beta, lambduh, X, y)
    iteration = 0
    while (iteration < max_iter):
        t = back_tracking(beta, lambduh, t_init, X, y)
        beta = beta - t * grad_beta

        # Store all of the places we step to
        beta_vals = np.vstack((beta_vals, beta))
        grad_beta = compute_grad(beta, lambduh, X, y)
        iteration = iteration + 1
        if (iteration % 200 == 0):
            print('gradient descent iteration', iteration)
    return beta_vals


# fast gradient descent algorithm
def fast_gradient(beta, theta, t_init, lambduh, X, y, max_iter=1000):
    grad_theta = compute_grad(theta, lambduh, X, y)
    beta_vals = beta
    iteration = 0
    while (iteration < max_iter):
        t = back_tracking(beta, lambduh, t_init, X, y)
        beta_new = theta - t * grad_theta
        theta = beta_new + iteration / (iteration + 3) * (beta_new - beta)

        # Store all of the places we step to
        beta_vals = np.vstack((beta_vals, beta_new))
        grad_theta = compute_grad(theta, lambduh, X, y)
        beta = beta_new
        iteration = iteration + 1
        if (iteration % 200 == 0):
            print('fast gradient descent iteration', iteration)
    return beta_vals


# plot the objective value of gradient and fast gradient algorithms
def plot_objs(betas_g, betas_fg, lambduh, X, y):
    length = betas_g.shape[0]
    objs_g = np.zeros(length)
    objs_fg = np.zeros(length)
    for i in range(length):
        objs_g[i] = compute_obj(betas_g[i, :], lambduh, X, y)
        objs_fg[i] = compute_obj(betas_fg[i, :], lambduh, X, y)

    fig = plt.figure(figsize=(12, 8))
    plt.plot(range(length), objs_g, c='blue', label='gradient')
    plt.plot(range(length), objs_fg, c='red', label='fast gradient')
    plt.xlabel('Number of Iteration')
    plt.ylabel('Objective Value')
    plt.title('Objective value vs. iteration when Lambda = ' + str(lambduh))
    plt.legend(loc='upper right')
    plt.show()


# compute misclassification error
def misclassification_error(beta_opt, X, y):
    y_pred = 1 / (1 + np.exp(-X.dot(beta_opt))) > 0.5

    # Convert to +/- 1
    y_pred = y_pred * 2 - 1
    return np.mean(y_pred != y)


# plot misclassification error
def plot_misclassification_error(betas_g, betas_fg, X, y):
    length = np.size(betas_g, 0)
    error_g = np.zeros(length)
    error_fg = np.zeros(length)
    for i in range(length):
        error_g[i] = misclassification_error(betas_g[i, :], X, y)
        error_fg[i] = misclassification_error(betas_fg[i, :], X, y)

    fig = plt.figure(figsize=(12, 8))
    plt.plot(range(length), error_g, c='blue', label='gradient')
    plt.plot(range(length), error_fg, c='red', label='fast gradient')
    plt.xlabel('Number of Iteration')
    plt.ylabel('Misclassification error')
    plt.title('Misclassification error vs. iteration when Lambda = ' + str(lambduh))
    plt.legend(loc='upper right')
    plt.show()


# cross validation to find the optimal lambuh with the smallest misclassification error
def find_optimal_lambduh(t, X, y):
    lambduhs = [10.0 ** i for i in np.arange(-10, 10, 1)]
    mis_errors = np.zeros(len(lambduhs))
    n, d = X.shape

    for i in range(len(lambduhs)):
        beta = np.zeros(d)
        theta = np.zeros(d)
        print("lambduh = " + str(lambduhs[i]))
        beta_vals = fast_gradient(beta, theta, t, lambduhs[i], X, y)
        mis_errors[i] = misclassification_error(beta_vals[-1], X, y)

    fig = plt.figure(figsize=(12, 8))
    plt.plot(np.log(lambduhs), mis_errors, c='red')
    plt.xlabel('log(lambduhs)')
    plt.ylabel('Misclassification Error')
    plt.title('Misclassification Errors for various lambda by SVM')

    print('Smallest Misclassification Error:', np.min(mis_errors), 'at lambda =', lambduhs[np.argmin(mis_errors)])
    return lambduhs[np.argmin(mis_errors)]


"""
When we execute the below codes, the above functions will be called.
There are three plots:
1. the objective value vs. iteration between gradient and fast gradient;
2. the misclassification error vs. iteration between gradient and fast gradient.
3. the misclassification error vs. log(lambda) by the fast gradient.
"""
X, y = simulated_data()
print("X: ", X.shape)
print("y: ", y.shape)

# Split the data into traning and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test: ", y_test.shape)

n, d = X_train.shape
lambduh = 1
t_init = 0.1

# run the gradient descent algorithm
beta = np.zeros(d)
betas_g = gradient(beta, t_init, lambduh, X_train, y_train)

# run the fast gradient descent algorithm
beta = np.zeros(d)
theta = np.zeros(d)
betas_fg = fast_gradient(beta, theta, t_init, lambduh, X_train, y_train)
plot_objs(betas_g, betas_fg, lambduh, X_train, y_train)
plot_misclassification_error(betas_g, betas_fg, X_train, y_train)
opt_lambduh = find_optimal_lambduh(t_init, X_train, y_train)