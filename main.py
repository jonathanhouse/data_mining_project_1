import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import argparse

class LinearRegressionModel:

    learning_rate = 1e-2
    slope_weights = np.random.rand(2)
    intercept_weight = np.random.rand(1)

    def __init__(self, learning_rate=1e-2):
        self.learning_rate = learning_rate
        self.mode_type = type

    def r2_score(self, y_true, y_pred):
        err2 = np.sum((y_true - y_pred) ** 2)
        var = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (err2 / var)

    def gradient_descent(self, X, y, test, iterations=int(1e8),label="Gradient Descent"):
        N = len(y)
        loss = []
        curr_r2 = 0

        while (curr_r2 <= 0) and (iterations > 0):
            predictions = self.predict(X)
            errors = y - predictions 
            dL_dm = -(2/N) * np.dot(X.T, errors)
            dL_db = -(2/N) * np.sum(errors)

            self.slope_weights -= self.learning_rate * dL_dm
            self.intercept_weight -= self.learning_rate * dL_db

            curr_r2_test = self.r2_score(test[:, -1], self.predict(test[:, :-1])) 
            curr_r2_train = self.r2_score(y, predictions)
            curr_r2 = min(curr_r2_test, curr_r2_train)
            iterations -= 1

            loss.append(np.dot(errors,errors)/N)
            # print(iterations, curr_r2_train, curr_r2_test, loss[-1])

        fig, ax = plt.subplots()
        ax.plot(loss)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('MSE')
        ax.set(title=f'MSE Loss over Time for {label} model (lr={self.learning_rate})')
        plt.show()

    def train(self, X, y, test=None, iterations=int(1e8),label="Gradient Descent"):
        self.slope_weights = np.random.rand(X.shape[1])
        self.intercept_weight = np.random.rand(1)
        self.gradient_descent(X, y, test=test, iterations=iterations,label=label)

    def predict(self, X):
        return np.dot(X, self.slope_weights) + self.intercept_weight

    def test(self, train, test):

        x_train = train[:,:-1].reshape(len(train),-1)
        y_train = train[:,-1]
        predict_train = self.predict(x_train)
        errors_train = y_train - predict_train
        mse_train = np.dot(errors_train,errors_train)/len(y_train)
        r2_train = self.r2_score(y_train, predict_train)

        x_test = test[:,:-1].reshape(len(test),-1)
        y_test = test[:,-1]
        predict_test = self.predict(x_test)
        errors_test = y_test - predict_test
        mse_test = np.dot(errors_test,errors_test)/len(y_test)
        r2_test = self.r2_score(y_test, predict_test)

        return mse_train, r2_train, mse_test, r2_test

    def summarize(self, train, test, df=None, label="Gradient Descent"):
        mse_train, r2_train, mse_test, r2_test = self.test(train,test)
        # print("Gradient Descent Summary:")
        # print(f"Train: MSE={mse_train}, R2 Score={r2_train}")
        # print(f"Test: MSE={mse_test}, R2 Score={r2_test}\n")
        if df is not None:
            df.loc[len(df)] = [label, mse_train, mse_test, r2_train, r2_test]

    def print_weights(self):
        print(f"Weights: slope={self.slope_weights}, intercept={self.intercept_weight}")

def import_data(file_path):
    data = pd.read_csv(file_path)
    return data


def train_test_split(df):
    indices = np.arange(df.shape[0])
    test_indices = (indices >= 501) & (indices <= 630)
    test = df.loc[test_indices,:]
    train = df.loc[~test_indices,:]
    return train, test

def min_max_normalize(data):
    col_min = data.min(axis=0, keepdims=True)
    col_max = data.max(axis=0, keepdims=True)
    norm_data = (data - col_min) / (col_max - col_min)
    return norm_data

def log_transform(data):
    return np.log(data + 1)


def run_gradient_descent(train,test,learning_rate=1e-7,iterations=int(1e8),df=None,label="Gradient Descent"):
    x = train[:, :-1].reshape((len(train),-1))
    y = train[:, -1]
    model = LinearRegressionModel(learning_rate=learning_rate)
    model.train(x, y, test=test, iterations=iterations, label=label)
    model.summarize(train, test, df=df, label=label)

def run_sk_linear_regression(train,test,df=None):
    x = train[:, :-1].reshape((len(train),-1))
    y = train[:, -1]

    model = LinearRegression().fit(x,y)
    score_train = model.score(x,y)
    score_test = model.score(test[:,:-1],test[:,-1])
    slopes = model.coef_
    intercept = model.intercept_
    train_preds = np.dot(train[:,:-1],slopes) + intercept
    errors_train = train[:,-1] - train_preds
    mse_train = np.dot(errors_train,errors_train)/len(train[:,-1])

    test_preds = np.dot(test[:,:-1],slopes) + intercept
    errors_test = test[:,-1] - test_preds
    mse_test = np.dot(errors_test,errors_test)/len(test[:,-1])

    # print("SK Linear Regression Summary")
    # print(f"Train: MSE = {mse_train}, R2 = {score_train}")
    # print(f"Test: MSE = {mse_test}, R2 = {score_test}\n")
    # print("Weights + Intercept: ", slopes,intercept)
    return model

def run_statsmodels_ols(X,Y):
    model = sm.OLS(Y,X).fit()
    return model

def statistical_significance_analysis(data):

    X_raw = data.iloc[:, :-1].copy()
    Y_raw = data.iloc[:, -1]

    X = sm.add_constant(X_raw, has_constant='add')
    Y = Y_raw

    # normalized data 
    X_norm_values = min_max_normalize(X_raw.to_numpy())
    Y_norm_values = min_max_normalize(Y_raw.to_numpy())
    X_norm_df = pd.DataFrame(
        X_norm_values,
        columns=X_raw.columns,
        index=X_raw.index
    )
    X_norm = sm.add_constant(X_norm_df, has_constant='add')
    Y_norm = pd.Series(
        Y_norm_values,
        index=Y_raw.index,
        name=Y_raw.name
    )

    # log transformed data 
    X_log_values = log_transform(X_raw.to_numpy())
    Y_log_values = log_transform(Y_raw.to_numpy())

    X_log_df = pd.DataFrame(
        X_log_values,
        columns=X_raw.columns,
        index=X_raw.index
    )

    X_log = sm.add_constant(X_log_df, has_constant='add')
    Y_log = pd.Series(
        Y_log_values,
        index=Y_raw.index,
        name=Y_raw.name
    )

    # create models 
    model = run_statsmodels_ols(X, Y)
    model_norm = run_statsmodels_ols(X_norm, Y_norm)
    model_log = run_statsmodels_ols(X_log, Y_log)

    # output dataframe 
    df = pd.DataFrame({
        'Feature': model.params.index,
        'P-value': model.pvalues.values,
        'P-Value (Normalized)': model_norm.pvalues.values,
        'P-Value (Log-Transformed)': model_log.pvalues.values
    })

    print(df)

def convergence_speed_analysis(train,test):

    train = train.to_numpy()
    test = test.to_numpy()

    num_iterations = int(1e5)
    train_norm = min_max_normalize(train)
    test_norm = min_max_normalize(test)

    df = pd.DataFrame(columns=['Model','MSE Train', 'MSE Test', 'R2 Train', 'R2 Test'])

    run_gradient_descent(train,test,iterations=num_iterations,df=df,label="Gradient Descent, Multivariate")
    run_gradient_descent(train_norm,test_norm,iterations=num_iterations,df=df,label="Normalized GD, Multivariate")

    print(df)

def nonlinear_analysis(train,test):

    train = train.to_numpy()
    test = test.to_numpy()

    num_iterations = int(1e5)
    df = pd.DataFrame(columns=['Model','MSE Train', 'MSE Test', 'R2 Train', 'R2 Test'])

    train_norm = min_max_normalize(train)
    test_norm = min_max_normalize(test)

    train_log = log_transform(train)
    test_log = log_transform(test)

    run_gradient_descent(train,test,iterations=num_iterations,df=df,label="Gradient Descent, Multivariate")
    run_gradient_descent(train_norm,test_norm,iterations=num_iterations,df=df,label="Normalized GD, Multivariate")
    run_gradient_descent(train_log,test_log,iterations=num_iterations,df=df,label="Log-Transformed GD")

    print(df)

def univariate_analysis(train,test):

    df = pd.DataFrame(columns=['Feature','Multivariate Coefficient','Univariate Coefficient'])
    df.loc[:,'Feature'] = train.columns[:-1]

    train = train.to_numpy()
    test = test.to_numpy()

    model = run_sk_linear_regression(train,test)

    slopes = model.coef_
    for i, slope in enumerate(slopes):
        df.loc[i, 'Multivariate Coefficient'] = slope

    for col in range(train.shape[1]-1):
        model = run_sk_linear_regression(train[:,[col,-1]],test[:,[col,-1]])
        df.loc[col, 'Univariate Coefficient'] = model.coef_[0]

    print(df)



def __main__():

    np.random.seed(0)
    argument_parser = argparse.ArgumentParser(description='Run linear regression analysis on concrete data.')
    argument_parser.add_argument('--analysis', type=str, choices=['convergence', 'nonlinear', 'univariate', 'statistical', 'all'], default='statistical', help='Type of analysis to run')
    data = import_data('Concrete_Data.csv')
    train, test = train_test_split(data)

    args = argument_parser.parse_args()
    if args.analysis == 'convergence':
        convergence_speed_analysis(train, test)
    elif args.analysis == 'nonlinear':
        nonlinear_analysis(train, test)
    elif args.analysis == 'univariate':
        univariate_analysis(train, test)
    elif args.analysis == 'statistical':
        statistical_significance_analysis(data)
    elif args.analysis == 'all':
        convergence_speed_analysis(train, test)
        nonlinear_analysis(train, test)
        univariate_analysis(train, test)
        statistical_significance_analysis(data)


if __name__ == "__main__":
    __main__()