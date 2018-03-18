from sklearn.externals import joblib

clf = joblib.load("./sample/dump/freq.pkl")

def detect_language(text):
    text = text.lower()
    code_a, code_z = (ord("a"), ord("z"))
    count = [0 for i in range(26)]
    for ch in text:
        n = ord(ch) - code_a
        if 0 <= n < 26: count[n] += 1
    total = sum(count)
    if total == 0: return "no input data"
    frequency = list(map(lambda n: n/total, count))
    predict = clf.predict([frequency])
    lang_dic = {"en": "English", "fr": "French", "id":"Indonesian Language", "tl" : "Tagalog"}
    return lang_dic[predict[0]]

text = input("enter any message to detect language\n")
if (text != ""):
    language = detect_language(text)
    print("outcome : ", language)