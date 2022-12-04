import json

train_set = open('../00_data/train_set.json', 'r').read()



lines = train_set.split('\n')
lines.pop()

categories = dict()

for line in lines:
    news = json.loads(line)
    if news['category'] not in categories:
        categories[news['category']] = 1


print(categories)




