import json
import random

random.seed(42)

train_set = open('train_set.json', 'w')
test_set = open('test_set.json', 'w')

data = open('../00_data/News_Category_Dataset_v3.json', 'r').read()
lines = data.split('\n')
lines.pop()

for line in lines:
    news = json.loads(line)
    item = {
        'headline': news['headline'],
        'category': news['category']
    }
    news = json.dumps(item)

    rand = random.random()
    if rand < 0.90:
        train_set.write(news + '\n')
    else:
        test_set.write(news + '\n')
