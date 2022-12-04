import json

train_set = open('train_set.json', 'r').read()

lines = train_set.split('\n')
lines.pop()

categories = dict()

for line in lines:
    news = json.loads(line)
    if news['category'] not in categories:
        categories[news['category']] = 1
    else:
        categories[news['category']] += 1

partial = 0
for category in categories:
    value = categories[category]
    categories[category] += partial
    partial += value

for category in categories:
    categories[category] /= partial

file = open('model.json', 'w')
file.write(json.dumps(categories))