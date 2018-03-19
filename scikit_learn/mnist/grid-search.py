import pandas
from sklearn import model_selection, svm, metrics
from sklearn.grid_search import GridSearchCV

train_csv = pandas.read_csv("./sample/train.csv")
test_csv = pandas.read_csv("./sample/t10k.csv")

train_label = train_csv.ix[:, 0]
train_data = train_csv.ix[:, 1:577]
test_label = test_csv.ix[:, 0]
test_data = test_csv.ix[:, 1:577]
print("the number of training data : ", len(train_label))

params = [
    {"C": [1, 10, 100, 10000], "kernel": ["linear"]},
    {"C": [1, 10, 100, 10000], "kernel": ["rbf"], "gamma": [0.001, 0.0001]},
]

classifier = GridSearchCV(svm.SVC(), params, n_jobs=-1)
classifier.fit(train_data, train_label)
print(classifier.best_estimator_)

predict = classifier.predict(test_data)
accuracy_score = metrics.accuracy_score(predict, test_label)
print("accuracy score : ", accuracy_score)
