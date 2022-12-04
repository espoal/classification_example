import json
import re

from category_mapper import category_mapper

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
STOPWORDS = set(stopwords.words('english'))

def datacleaning(text):


    whitespace = re.compile(r"\s+")
    user = re.compile(r"(?i)@[a-z0-9_]+")
    text = whitespace.sub(' ', text)
    text = user.sub('', text)
    text = re.sub(r"\[[^()]*\]", "", text)
    text = re.sub("\d+", "", text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r"(?:@\S*|#\S*|http(?=.*://)\S*)", "", text)

    #text = text.lower()


    # removing stop-words
    text = [word for word in text.split() if word not in list(STOPWORDS)]

    # word lemmatization
    sentence = []
    for word in text:
        lemmatizer = WordNetLemmatizer()
        sentence.append(lemmatizer.lemmatize(word, 'v'))

    return ' '.join(sentence)


train_set = open('../data/train_set.json', 'r').read()



lines = train_set.split('\n')
lines.pop()

categories = dict()

for line in lines:
    news = json.loads(line)
    real_category = category_mapper[news['category']]
    real_headline = news['headline']
    #real_headline = datacleaning(news['headline']) #

    if real_category not in categories:
        categories[real_category] = dict()

    category = categories[real_category]
    words = real_headline.split(' ')
    for word in words:
        if word not in category:
            category[word] = 1
        category[word] += 1

category_iterator = iter(categories)

intersection = categories[next(category_iterator)].keys()

for category in category_iterator:
    next_category = categories[category].keys()
    intersection = intersection & next_category

category_iterator = iter(categories)

for category in category_iterator:
    next_category = categories[category]
    for word in intersection:
        del next_category[word]



file = open('model.json', 'w')
file.write(json.dumps(categories))
