# classification_example


## Step 0: Data analysis

First step was to download the data, look at the paper, try
to understand the data and its shape. 

To reproduce this step make sure the file `News_Category_Dataset_v3.json`
is saved in the `00_data` directory. Unfortunately this can't be
automated because the download requires a login.


## Step 1: Random model

The first step was to create a random model, to have a baseline.
The performance is quite bad, around 1e-3, which is not surprising
but still serves as a point of reference.

How to run:
```bash
source venv/bin/activate
cd 01_random
python3 prep.py # creates the data
python3 train.py # trains the model
python3 infer.py # evaluates the model
```

## Step 2: Bag of word model

The second step was to create a simple bag of word model. We create an hashmap
storing the words used in each category as model, an headline is then classified
by finding the category with the highest number of words in common.

Instead of hashmaps I could have use bloom filters, making the model weight only 128 bytes
(32 UTF-8 characters) in total, but I didn't to save time. The model is very fast, both
in training and inference, faster than any other (except the random model) by several orders of magnitude.

The only cleaning of the data I did was to remove common words, using a simple intersection of dictionaries.

**The accuracy is around 38.61%.**

Accuracy gets greatly boosted at **60.92% by using category aggregation** (see later). I'm sure with more
time I could squeeze another 10% out of it, by using digrams, better correlation matrices, ...


How to run:
```bash
source venv/bin/activate
cd 01_bloom
python3 prep.py # creates the data
python3 train.py # trains the model
python3 infer.py # evaluates the model
```

## Step 3: Tensorflow model

The third step was to follow a tensorflow tutorial. The model is much
more expensive than the previous one, but gives around 10% more accuracy.

I tried several models (binary, non-binary, BERT, ...) and in the end I
picked the binary model because it's cheaper than the others and has very
good performance, at 

As with the previous model using category aggregation gives a nice boost in accuracy,
**at 69.92%**.

How to run:
```bash
source venv/bin/activate
cd 02_tf
python3 prep.py # creates the data
python3 train.py # trains and evaluates the model
```

## Step 4: Categories aggregation

After trying several models and analyzing the errors, I noticed that the
categories were too difficult to distinguish. Using a correlation matrix
and principal component analysis one could find categories that are very
similar, and aggregate them. 

This then becomes an optimization problem: if we pick only a category (news)
then our prediction power becomes 100%, but we have no significance. If we don't
aggregate at all then we have 100% significance but prediction powers tops at 55%.

[Here](./04_projection/category_mapper.py) I propose a possible aggregation,
but unfortunately I was already at the 5th hour so time was running short, and
I couldn't work on it as much as I wanted.


## Step 5: Google scholar

Another I approach I investigated but didn't have time to implement was to
reverse search the paper associated with the data on google scholar.

Here's a nice result:
https://proceedings.neurips.cc/paper/2021/file/8493eeaccb772c0878f99d60a0bd2bb3-Paper.pdf

## Step 6: Commercial solutions

Another approach I investigated was to use commercial solutions, like Vertex AI or
AWS Comprehend. Unfortunately I didn't have time to implement them, but I'm sure
they would have had interesting performance at the cost of losing ownership of the
solution.