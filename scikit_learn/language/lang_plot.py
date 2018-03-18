import matplotlib.pyplot as pyplot
import pandas
import lang_train

train_data = lang_train.load_files("./sample/train/*.txt")
train_data_frequencies = train_data["frequencies"]
train_data_labels = train_data["labels"]

language_dic = {}
for i, language_label in enumerate(train_data_labels):
    frequency = train_data_frequencies[i]
    if not (language_label in language_dic):
        language_dic[language_label] = frequency
        continue
    for idx, frequency_per_code in enumerate(frequency):
        language_dic[language_label][idx] = (language_dic[language_label][idx] + frequency_per_code) / 2

asclist = [[chr(n) for n in range(97, 97+26)]]
df = pandas.DataFrame(language_dic, index=asclist)

pyplot.style.use('ggplot')
df.plot(kind="bar", subplots=True, ylim=(0, 0.15))
pyplot.show()
