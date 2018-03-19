from sklearn import model_selection, svm, metrics

def load_csv(file_name):
    labels = []
    images = []
    with open(file_name, "r") as file:
        for line in file:
            cols = line.split(",")
            if len(cols) < 2 : continue
            labels.append(int(cols.pop(0)))
            # how to make data into vector
            # pixel(0~255) / 256 -> 0 ~ 1
            function = lambda  n : int(n) / 256
            vals = list(map(function, cols))
            images.append(vals)
    return {"labels" : labels, "images" : images}

train_data = load_csv("./sample/train.csv")
test_data = load_csv("./sample/t10k.csv")

# training
classifier = svm.SVC()
classifier.fit(train_data["images"], train_data["labels"])

# prediction
predict = classifier.predict(test_data["images"])

# evalutation
accuray_score = metrics.accuracy_score(test_data["labels"], predict)
cl_report = metrics.classification_report(test_data["labels"], predict)
print("accuracy score : ", accuray_score)
print("report :")
print(cl_report)