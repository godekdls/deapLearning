from sklearn import svm, metrics
import glob, os.path, re, json

def check_frequency(file_name):
    name = os.path.basename(file_name)
    language = re.match(r'^[a-z]{2,}', name).group()
    with open(file_name, "r", encoding="utf-8") as file:
        text = file.read()
    text = text.lower()
    count = [0 for n in range(0, 26)]
    code_a = ord("a")
    code_z = ord("z")
    for ch in text:
        n = ord(ch)
        if (code_a <= n <= code_z): # between a~z
            count[n - code_a] += 1
    total = sum(count)
    frequency = list(map(lambda n: n / total, count))
    return (frequency, language)

def load_files(path):
    frequencies = []
    labels = []
    file_list = glob.glob(path)
    for file_name in file_list:
        frequency = check_frequency(file_name)
        frequencies.append(frequency[0])
        labels.append(frequency[1])
    return {"frequencies" : frequencies, "labels" : labels}

if __name__ == "__main__":
    train_data = load_files("./sample/train/*.txt")
    test_data = load_files("./sample/test/*.txt")

    # training
    classifier = svm.SVC()
    classifier.fit(train_data["frequencies"], train_data["labels"])

    # prediction
    predict = classifier.predict(test_data["frequencies"])

    # evaluation
    accuracy_score = metrics.accuracy_score(test_data["labels"], predict)
    classification_report = metrics.classification_report(test_data["labels"], predict)
    print("accuracy socre : ", accuracy_score)
    print("report : ")
    print(classification_report)