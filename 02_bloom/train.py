import json

from category_mapper import category_mapper

use_aggregations = True

train_set = open('train_set.json', 'r').read()
lines = train_set.split('\n')
lines.pop()

categories = dict()

# Build the word dictionary
for line in lines:
    news = json.loads(line)
    if use_aggregations:
        real_category = category_mapper[news['category']]
    else:
        real_category = news['category']
    real_headline = news['headline']

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
