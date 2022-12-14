import os
import json
import random
import shutil
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from category_mapper import category_mapper


# Set to false if you don't want to use categories aggregations
use_aggregations = True

random.seed(42)

train_set = []
test_set = []

data = open('../00_data/News_Category_Dataset_v3.json', 'r').read()
lines = data.split('\n')
lines.pop()

STOPWORDS = set(stopwords.words('english'))

# Data cleaning functions copied from another notebook
def datacleaning(text):

    whitespace = re.compile(r"\s+")
    user = re.compile(r"(?i)@[a-z0-9_]+")
    text = whitespace.sub(' ', text)
    text = user.sub('', text)
    text = re.sub(r"\[[^()]*\]", "", text)
    text = re.sub("\d+", "", text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r"(?:@\S*|#\S*|http(?=.*://)\S*)", "", text)

    text = text.lower()


    # removing stop-words
    text = [word for word in text.split() if word not in list(STOPWORDS)]

    # word lemmatization
    sentence = []
    for word in text:
        lemmatizer = WordNetLemmatizer()
        sentence.append(lemmatizer.lemmatize(word, 'v'))

    return ' '.join(sentence)

for line in lines:
    news = json.loads(line)
    if use_aggregations:
        real_category = category_mapper[news['category']]
    else:
        real_category = news['category']
    item = {
        'headline': datacleaning(news['headline']),
        'category': real_category
    }

    rand = random.random()
    if rand < 0.95:
        train_set.append(item)
    else:
        test_set.append(item)



def prep_data(dataset, type):

    train_path = "/tmp/classification/" + type + "/"
    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    os.makedirs(train_path)


    categories = dict()

    for news in dataset:
        category = news['category']
        category_path = train_path + category + '/'
        if category not in categories:
            categories[category] = 0
            os.mkdir(category_path)
        idx = categories[category]
        categories[category] += 1
        f = open(category_path + str(idx) + '.txt', 'w')
        f.write(news['headline'])


prep_data(train_set, 'train')


prep_data(test_set, 'test')