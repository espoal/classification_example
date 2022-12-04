import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import utils
from tensorflow.keras.layers import TextVectorization


use_aggregations = False


batch_size = 32
seed = 42

path = '/tmp/classification/'

raw_train_ds = utils.text_dataset_from_directory(
    path + 'train/',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed
)

raw_val_ds = utils.text_dataset_from_directory(
    path + 'train/',
    batch_size=batch_size,
    validation_split=.2,
    subset='training',
    seed=seed
)

raw_test_ds = utils.text_dataset_from_directory(
    path + 'test/',
    batch_size=batch_size
)

VOCAB_SIZE = 10000

binary_vectorize_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='binary')

MAX_SEQUENCE_LENGTH = 250



train_text = raw_train_ds.map(lambda text, labels: text)
binary_vectorize_layer.adapt(train_text)

def binary_vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return binary_vectorize_layer(text), label


binary_train_ds = raw_train_ds.map(binary_vectorize_text)
binary_val_ds = raw_val_ds.map(binary_vectorize_text)
binary_test_ds = raw_test_ds.map(binary_vectorize_text)


AUTOTUNE = tf.data.AUTOTUNE

def configure_dataset(dataset):
    return dataset.cache().prefetch(buffer_size=AUTOTUNE)


binary_train_ds = configure_dataset(binary_train_ds)
binary_val_ds = configure_dataset(binary_val_ds)
binary_test_ds = configure_dataset(binary_test_ds)

if use_aggregations:
    labels_count = 12
else:
    labels_count = 42

binary_model = tf.keras.Sequential([layers.Dense(labels_count)])

binary_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])

history = binary_model.fit(
    binary_train_ds, validation_data=binary_val_ds, epochs=10)

print("Linear model on binary vectorized data:")
print(binary_model.summary())

binary_loss, binary_accuracy = binary_model.evaluate(binary_test_ds)

print("Binary model accuracy: {:2.2%}".format(binary_accuracy))

