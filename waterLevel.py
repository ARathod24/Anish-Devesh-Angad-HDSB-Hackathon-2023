import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import keras as K

test_results = {}

np.set_printoptions(precision=3, suppress=True)


ds_train = pd.read_csv('waterLevelTraining.csv')
ds_test = pd.read_csv('waterLevelTesting.csv')

desired_feature = ['water_level']

X_train = ds_train.drop(desired_features, axis = 1)
y_train = ds_train[desired_features]

X_test = ds_test.drop(desired_features, axis = 1)
y_test = ds_test[desired_features]

train_features = X_train.copy()
test_features = X_test.copy()


X_train.describe().transpose()[['mean', 'std']]

normalizer = tf.keras.layers.Normalization(axis=-1)

normalizer.adapt(np.array(X_train))

print(normalizer.mean.numpy())

first = np.array(X_train[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())


def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [Predicted Vals]')
  plt.legend()
  plt.grid(True)


def build_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='elu'),
      layers.Dense(64, activation='elu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.01),
                #optimizer = K.optimizers.SGD(0.01)
                )
  return model

dnn_model = build_model(normalizer)
dnn_model.summary()


history = dnn_model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    verbose=0, epochs=1000)

final_loss = history.history['loss'][-1]
final_validation_loss = history.history['val_loss'][-1]

plot_loss(history)
plt.show()
print(final_loss)
print(final_validation_loss)

test_results['dnn_model'] = dnn_model.evaluate(X_test, y_test, verbose=1)

pd.DataFrame(test_results, index=['Mean absolute error [predicted_features]']).T

test_predictions = dnn_model.predict(test_features).flatten()

dnn_model.save('dnn_model')

