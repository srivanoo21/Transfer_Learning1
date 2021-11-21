import tensorflow as tf
import os
import numpy
import time
import numpy as np

def get_timestamp(name):
    timestamp = time.asctime().replace(" ", "_").replace(":", "_")
    unique_name = f"{name}_at_{timestamp}"
    return unique_name

def get_callbacks(config, X_train):
    logs = config['logs']
    unique_dir_name = get_timestamp('tb_logs')
    TENSORBOARD_ROOT_LOG_DIR = os.path.join(logs['logs_dir'], logs['tensorboard_root_log_dir'], unique_dir_name) 

    os.makedirs(TENSORBOARD_ROOT_LOG_DIR, exist_ok=True)

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_ROOT_LOG_DIR)

    file_writer = tf.summary.create_file_writer(logdir=TENSORBOARD_ROOT_LOG_DIR)

    with file_writer.as_default():
        images = np.reshape(X_train[10:30], (-1, 28, 28, 1)) ### <<< 20, 28, 28, 1
        tf.summary.image("20 handritten digit samples", images, max_outputs=25, step=0)

    params = config["params"]
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        patience=params["patience"], 
        restore_best_weights=params["restore_best_weights"])

    artifacts = config["artifacts"]
    CKPT_DIR = os.path.join(artifacts["artifacts_dir"], artifacts["checkpoint_dir"])
    os.makedirs(CKPT_DIR, exist_ok=True)
    
    CKPT_path = os.path.join(CKPT_DIR, params["model_ckpt"])

    checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(CKPT_path, save_best_only=True)

    return [tensorboard_cb, early_stopping_cb, checkpointing_cb]
