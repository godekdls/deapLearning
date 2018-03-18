import pandas
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

# https://github.com/pandas-dev/pandas/blob/master/pandas/tests/data/iris.csv
csv = pandas.read_csv('./iris.csv')

csv_data = csv[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]
csv_label = csv["Name"]
# split data into training data(75%) and test data(25%)
train_data, test_data, train_label, test_label = train_test_split(csv_data, csv_label)

# training
clf = svm.SVC()  # use SVM algorithm
clf.fit(train_data, train_label)

# prediction
pre = clf.predict(test_data)

# evaluation
ac_score = metrics.accuracy_score(test_label, pre)
print("accuracy score : ", ac_score)