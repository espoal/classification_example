import json
import random

random.seed(42)

model_file = open('model.json', 'r').read()
model = dict(json.loads(model_file))

test_file = open('../data/validation_set.json', 'r').read()
test_set = test_file.split('\n')
test_set.pop()

correct = 0
wrong = 0

for line in test_set:
    news = json.loads(line)
    oracle = random.random()
    prev = 0.0
    prevision = ""
    for category in model:
        value = model[category]
        if (prev < oracle) & (value < oracle):
            prevision = category
            break
        prev = value

    if prevision == news['category']:
        correct += 1
    else:
        wrong += 1


print("Correct: ", correct)
print("Wrong: ", wrong)