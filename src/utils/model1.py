import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import io
import logging


def create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES):

    # set the seeds 
    seed = 2021  # put it in config
    tf.random.set_seed(seed)
    np.random.seed(seed)


    # log our model summary information in logs
    def _log_model_summary(model):
        with io.StringIO() as stream:
            model.summary(print_fn= lambda x: stream.write(f"{x}\n"))
            summary_str = stream.getvalue()
        return summary_str


def get_unique_filename(filename):
    unique_filename = time.strftime(f"%Y%m%d_%H%M%S_{filename}")
    return unique_filename


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