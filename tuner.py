import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

print(x_train.shape)

print(set(y_train))

def create_model(hp):
  num_hidden_layers = 1
  num_units = 8
  dropout_rate = 0.1
  learning_rate = 0.01

  if hp:
    num_hidden_layers = hp.Choice('num_hidden_layers', values = [1,2,3])
    num_units = hp.Choice('num_units', values = [8, 16, 32])
    dropout_rate = hp.Float('dropout_rate', min_value = 0.1, max_value = 0.5)
    learning_rate = hp.Float('learning_rate', min_value = 0.0001, max_value = 0.01)

  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Flatten(input_shape = (28,28) ))
  model.add(tf.keras.layers.Lambda(lambda x: x/255))

  for _ in range(0, num_hidden_layers):
    model.add(tf.keras.layers.Dense(num_units, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))

  model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

  model.compile(loss = 'sparse_categorical_crossentropy',
                optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
                metrics = ['accuracy'])
  return model

create_model(None).summary()

class CustomTuner(kt.tuners.BayesianOptimization):
  def run_trial(self, trial, *args, **kwargs):
    kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 32, 128, step = 32)
    return super(CustomTuner, self).run_trial(trial, *args, **kwargs)
  
tuner = CustomTuner(
    create_model,
    objective = 'val_accuracy',
    max_trials = 20,
    directory = 'logs',
    project_name = 'fashion_mnist',
    overwrite = True)

tuner.search_space_summary()

tuner.search(x_train, y_train, validation_data = (x_test, y_test),
             epochs = 5, verbose = False)

model = tuner.get_best_models(num_models = 1)[0]
model.summary()

model.fit(x_train, y_train, validation_data = (x_test, y_test),
    epochs = 20, batch_size = 128,
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', patience = 3)]
    )