import re
from collections import Counter
import numpy as np
from wordcloud import WordCloud
from nltk.corpus import stopwords
import os
import matplotlib.pyplot as plt


def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.pdf':
                data = os.path.splitext(file)[0]
                data = re.sub('_', ' ', data)
                data = re.sub('ICCV', '', data)
                data = re.sub('2019', '', data)
                data = re.sub('paper', '', data)
                data = re.sub(r'^\w+\s*', '', data)
                L.append(data)
    return L


def create_text(text_dir, list):
    file = open(text_dir, 'w')
    for line in list:
        file.write(line + '\n')
    file.close()


file_dir = './data/ICCV2019'
text_dir = './name.text'
# L = file_name(file_dir)
# create_text(text_dir, L)


text = open(text_dir).read()
content = text
hist = {}
data = []
stopwords_deep_learning = ['learning', 'network', 'neural', 'networks', 'deep', 'via', 'using',
                           'convolutional', 'single', 'image', 'object']
words = content.split()
paper = content.split('\n')
word_list = []
for word in words:
    word = word.lower()
    if word not in stopwords.words('english') and word not in stopwords_deep_learning:
        word_list.append(word)
for i in range(len(word_list)):
    if word_list[i] in hist:
        hist[word_list[i]] = hist[word_list[i]] + 1
    else:
        hist[word_list[i]] = 1
for key, value in hist.items():
    temp = [value, key]
    data.append(temp)
data.sort(reverse=True)
plt.rcParams['font.sans-serif'] = ['SimHei']


keywords = Counter(word_list)
for letter, count in keywords.most_common(15):
    print('%s: %4d' % (letter, count))
threshold = 25
keywords_mc = keywords.most_common(threshold)

plt.rcdefaults()
fig, ax = plt.subplots(figsize=(5, 6))

key = [k[0] for k in keywords_mc]
value = [k[1] for k in keywords_mc]
y_pos = np.arange(len(key))
ax.barh(y_pos, value, align='center', color='green', ecolor='black', log=True)
ax.set_yticks(y_pos)
ax.set_yticklabels(key, rotation=0, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('Frequency')
ax.set_title('ICCV 2019 Submission Top {} Keywords'.format(threshold))
plt.show()


wordcloud = WordCloud(max_font_size=64, max_words=160,
                      width=1280, height=640,
                      background_color="black").generate(' '.join(word_list))
plt.figure(figsize=(16, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


def findpapers(topic):
    for i in range(len(paper)):
        if topic in paper[i].lower():
            print(paper[i])


findpapers('detection')