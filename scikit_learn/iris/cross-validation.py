import pandas
from sklearn import svm, metrics, model_selection

csv = pandas.read_csv('./sample/iris.csv')

train_data = csv[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]
label = csv["Name"]

classifier = svm.SVC()
scores = model_selection.cross_val_score(classifier, train_data, label, cv=5)
print("each accuracy rate : ", scores)
print ("average accuracy rate : ", scores.mean())