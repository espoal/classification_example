import json
from train import datacleaning

from category_mapper import category_mapper

model_file = open('model.json', 'r').read()
model = json.loads(model_file)

test_file = open('../data/test_set.json', 'r').read()
test_set = test_file.split('\n')
test_set.pop()

correct = 0
wrong = 0

for line in test_set:
    news = json.loads(line)
    real_category = category_mapper[news['category']]
    words = news['headline'].split(' ')
    categories = dict()
    for category in model:
        categories[category] = 0
        for word in words:
            if word in model[category]:
                categories[category] += model[category][word]
        #categories[category] /= normalization[category]
    category = max(categories, key=categories.get)
    if category == real_category:
        correct += 1
    else:
        wrong += 1


print('Correct: ' + str(correct))
print('Wrong: ' + str(wrong))