import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from tensorflow.keras import regularizers

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format


def create_model(my_learning_rate, my_feature_layer):
    """Create and compile a simple linear regression model."""

    model = tf.keras.models.Sequential()

    model.add(my_feature_layer)

    # Define the first hidden layer with 20 nodes.
    model.add(tf.keras.layers.Dense(units=20,
                                    activation='softmax',
                                    name='Hidden1',
                                    activity_regularizer=regularizers.l2(0.01)
                                    ))

    # Define the output layer.
    model.add(tf.keras.layers.Dense(units=1,
                                    name='Output'))

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy',
                                                           threshold=classification_threshold)])

    return model


def train_model(model, dataset, epochs, label_name,
                batch_size=None):
    """Train the model by feeding it data."""

    # Split the dataset into features and label.
    features = {name: np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x=features, y=label, batch_size=batch_size,
                        epochs=epochs, shuffle=True)

    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch

    # To track the progression of training, gather a snapshot
    # of the model's mean squared error at each epoch.
    hist = pd.DataFrame(history.history)

    return epochs, hist


def plot_curve(epochs, hist, list_of_metrics):
    """Plot a curve of one or more classification metrics vs. epoch."""
    # list_of_metrics should be one of the names shown in:
    # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    # for m in list_of_metrics:
    x = hist["accuracy"]
    plt.plot(epochs[1:], x[1:], label="accuracy")

    plt.show()


# Some preprocessing, shuffling the data of test and training, also replacing M, B with 1,0

data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")
data_train = data_train.reindex(np.random.permutation(data_train.index))
data_test = data_test.reindex(np.random.permutation(data_test.index))

data_train['diagnosis'].replace('B', 0, inplace=True)
data_train['diagnosis'].replace('M', 1, inplace=True)

data_test['diagnosis'].replace('B', 0, inplace=True)
data_test['diagnosis'].replace('M', 1, inplace=True)

# finding the good features.
correlation_matrix = data_train.corr()

# %%

# NORMALIZE THE VALUES FIRST TO MAKE TRAINING EASIER
data_train_mean = data_train.mean()
data_train_std = data_train.std()
data_train_norm = (data_train - data_train_mean) / data_train_std

data_train_norm["diagnosis"] = (data_train_norm["diagnosis"] > 0).astype(float)

data_test_mean = data_test.mean()
data_test_std = data_test.std()
data_test_norm = (data_test - data_test_mean) / data_test_std

data_test_norm["diagnosis"] = (data_test_norm["diagnosis"] > 0).astype(float)

# choose the features and put them into a feature layer
feature_columns = []
for item in list(data_train_norm.columns):
    # remove useless features
    if item != "id" and item != "diagnosis" and item != "fractional_dimension_mean" and item != "texture_se" and item != "smoothness_se" and item != "symmetry_se" and item != "fractional_dimension_se" and item != "symmetry_se" and item != "Unnamed: 32":
        tr = tf.feature_column.numeric_column(item)
        feature_columns.append(tr)

feature_layer = layers.DenseFeatures(feature_columns)
feature_layer(dict(data_train_norm))

tf.keras.backend.clear_session()  # clear any existing model
learning_rate = 0.001
epochs = 50
batch_size = 5
label_name = "diagnosis"
classification_threshold = 0.65

# Establish the model's topography.
my_model = create_model(learning_rate, feature_layer)

# Train the model on the training set.
epochs, hist = train_model(my_model, data_train_norm, epochs,
                           label_name, batch_size)

plot_curve(epochs, hist, [tf.keras.metrics.BinaryAccuracy(name='accuracy',
                                                          threshold=classification_threshold)])

features = {name: np.array(value) for name, value in data_test_norm.items()}
label = np.array(features.pop(label_name))

my_model.evaluate(x=features, y=label, batch_size=batch_size)
