from sklearn import svm
from sklearn.externals import joblib
import os, os.path
import lang_train

train_data = lang_train.load_files("./sample/train/*.txt")

# training
clf = svm.SVC()
clf.fit(train_data["frequencies"], train_data["labels"])

# save trained data
save_path = "./sample/dump"
if not os.path.exists(save_path): os.mkdir(save_path)
joblib.dump(clf, save_path + "/freq.pkl")
print("saved")