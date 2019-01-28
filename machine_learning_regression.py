from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,SGDRegressor,Ridge
from sklearn.metrics import mean_squared_error


#正规方程预测波斯顿房价数据集
def ne_linear():
    boston = load_boston()

    x_train, x_test, y_train, y_test = train_test_split(boston.data,boston.target,random_state=22)

    transfer = StandardScaler()

    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = LinearRegression()
    estimator.fit(x_train,y_train)

    print("coef: \n",estimator.coef_)
    print("intercept: \n",estimator.intercept_)

    y_predict = estimator.predict(x_test)
    print("y_predict: \n",y_predict)

    error = mean_squared_error(y_test,y_predict)

    print("mse: \n", error)

    return None


#梯度下降预测波斯顿房价数据集
def gr_linear():
    boston = load_boston()

    x_train, x_test, y_train, y_test = train_test_split(boston.data,boston.target,random_state=22)

    transfer = StandardScaler()

    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = SGDRegressor()
    estimator.fit(x_train,y_train)

    print("coef: \n",estimator.coef_)
    print("intercept: \n",estimator.intercept_)

    y_predict = estimator.predict(x_test)
    print("y_predict: \n",y_predict)

    error = mean_squared_error(y_test,y_predict)

    print("mse: \n", error)

    return None


#岭回归预测波斯顿房价数据集
def ridge_linear():
    boston = load_boston()

    x_train, x_test, y_train, y_test = train_test_split(boston.data,boston.target,random_state=22)

    transfer = StandardScaler()

    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = Ridge(alpha=0.5,max_iter=1000)
    estimator.fit(x_train,y_train)

    print("coef: \n",estimator.coef_)
    print("intercept: \n",estimator.intercept_)

    y_predict = estimator.predict(x_test)
    print("y_predict: \n",y_predict)

    error = mean_squared_error(y_test,y_predict)

    print("mse: \n", error)

    return None


if __name__ == "__main__":
    #ne_linear()
    #gr_linear()
    ridge_linear()