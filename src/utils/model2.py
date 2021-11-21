import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import logging
import io


def create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES):

    # set the seeds 
    seed = 2021  # put it in config
    tf.random.set_seed(seed)
    np.random.seed(seed)


    LAYERS = [
          tf.keras.layers.Flatten(input_shape=[28, 28], name="inputLayer"),
          tf.keras.layers.Dense(300, name="hiddenLayer1"),
          tf.keras.layers.LeakyReLU(name="leakyrelu1"),
          tf.keras.layers.Dense(100, name="hiddenLayer2"),
          tf.keras.layers.LeakyReLU(name="leakyrelu2"),
          tf.keras.layers.Dense(2, activation="softmax", name="outputLayer")
            ]

    # Compile the model
    scratch_model = tf.keras.models.Sequential(LAYERS)
    scratch_model.compile(loss=LOSS_FUNCTION,
                  optimizer=OPTIMIZER,
                  metrics=METRICS)
    logging.info(">>>>>>>model is compiled <<<<<<<")

    # log our model summary information in logs
    def _log_model_summary(model):
        with io.StringIO() as stream:
            model.summary(print_fn= lambda x: stream.write(f"{x}\n"))
            summary_str = stream.getvalue()
        return summary_str

    # model_clf.summary()
    logging.info(f"scratch model summary: \n{_log_model_summary(scratch_model)}")

    return scratch_model ## untrained model


def get_unique_filename(filename):
    unique_filename = time.strftime(f"%Y%m%d_%H%M%S_{filename}")
    return unique_filename


def save_model(model, model_name, model_dir):
    #unique_filename = get_unique_filename(model_name)
    unique_filename = "scratch_model.h5"
    path_to_model = os.path.join(model_dir, unique_filename)
    model.save(path_to_model)


def save_plot(df, plot_name, plot_dir):
    unique_plotname = get_unique_filename(plot_name)
    path_to_plot = os.path.join(plot_dir, unique_plotname)
    df.plot(figsize=(10, 7))
    plt.grid(True)
    plt.savefig(path_to_plot)
