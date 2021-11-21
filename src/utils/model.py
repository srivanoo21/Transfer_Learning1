import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt

def create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES):

    LAYERS = [tf.keras.layers.Flatten(input_shape=[28, 28], name="inputLayer"),
          tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
          tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
          tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="outputLayer")]

    model_clf = tf.keras.models.Sequential(LAYERS)
    model_clf.summary()
    model_clf.compile(loss=LOSS_FUNCTION,
                  optimizer=OPTIMIZER,
                  metrics=METRICS)

    return model_clf ## untrained model


def get_unique_filename(filename):
    unique_filename = time.strftime(f"%Y%m%d_%H%M%S_{filename}")
    return unique_filename


def save_model(model, model_name, model_dir):
    unique_filename = get_unique_filename(model_name)
    path_to_model = os.path.join(model_dir, unique_filename)
    model.save(path_to_model)


def save_plot(df, plot_name, plot_dir):
    unique_plotname = get_unique_filename(plot_name)
    path_to_plot = os.path.join(plot_dir, unique_plotname)
    df.plot(figsize=(10, 7))
    plt.grid(True)
    plt.savefig(path_to_plot)


def get_log_path(log_name, logs_dir):
  uniqueName = get_unique_filename(log_name)
  logs_dir = os.path.join(logs_dir, log_name)
  log_path = os.path.join(logs_dir, uniqueName)
  print(f"savings logs at: {log_path}")

  return log_path