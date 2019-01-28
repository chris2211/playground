from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier


#knn分类iris数据集
def knn_iris():
    iris = load_iris()
    x_train, x_test,y_train,y_test = train_test_split(iris.data,iris.target,random_state=6)

    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train,y_train)

    y_predict = estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("compare:\n",y_test == y_predict)

    score = estimator.score(x_test,y_test)
    print("score:\n",score);
    return None

# 网格搜索  交叉验证
def knn_iris_gscv():
    iris = load_iris()
    x_train, x_test,y_train,y_test = train_test_split(iris.data,iris.target,random_state=6)

    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = KNeighborsClassifier()

    param_dict = {"n_neighbors": [1,3,5,7,9,11]}
    estimator = GridSearchCV(estimator,param_grid=param_dict,cv=10)


    estimator.fit(x_train,y_train)

    y_predict = estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("compare:\n",y_test == y_predict)

    score = estimator.score(x_test,y_test)
    print("score:\n",score)

    print("best param:\n",estimator.best_params_)
    print("best_score:\n",estimator.best_score_)
    print("best_estimator:\n",estimator.best_estimator_)
    print("best_result:\n",estimator.cv_results_)

    return None

#朴素贝叶斯
def nb_news():
    news = fetch_20newsgroups(subset="all")
    x_train,x_test,y_train,y_test = train_test_split(news.data,news.target)

    transfer = TfidfVectorizer()

    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = MultinomialNB()
    estimator.fit(x_train.x_test)

    y_predict = estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("compare:\n",y_test == y_predict)

    score = estimator.score(x_test,y_test)
    print("score:\n",score);
    return None

#决策树分类iris数据集
def decesion_iris():
    iris = load_iris()

    x_train, x_test, y_train, y_test = train_test_split(iris.data,iris.target,random_state=22)

    estimator = DecisionTreeClassifier(criterion="entropy")
    estimator.fit(x_train,y_train)

    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("compare:\n", y_test == y_predict)

    score = estimator.score(x_test, y_test)
    print("score:\n", score);

    return None

if __name__ == "__main__":
    #knn_iris()
    #knn_iris_gscv()
    #nb_news()
    decesion_iris()