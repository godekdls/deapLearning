import pandas as pd
from sklearn import svm, metrics

# xor data to try to train
xor_data = [
    # P, Q, result
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
]

# init data and label for training
data_frame = pd.DataFrame(xor_data)
data = data_frame.ix[:,0:1] # [ [0, 0], [0, 1], [1, 0], [1, 1] ]
label = data_frame.ix[:,2] # [0, 1, 1, 0]

# training
classifier = svm.SVC() # use SVM algorithm
classifier.fit(data, label)

# prediction
pre = classifier.predict(data)
print("prediction result: ", pre)

# evaluation
ac_score = metrics.accuracy_score(label, pre)
print("correct answer rate : ", ac_score)