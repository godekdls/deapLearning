from sklearn import svm
from sklearn.externals import joblib
import os, os.path
import lang_train

train_data = lang_train.load_files("./sample/train/*.txt")

# training
classifier = svm.SVC()
classifier.fit(train_data["frequencies"], train_data["labels"])

# save trained data
save_path = "./sample/dump"
if not os.path.exists(save_path): os.mkdir(save_path)
joblib.dump(classifier, save_path + "/freq.pkl")
print("saved")