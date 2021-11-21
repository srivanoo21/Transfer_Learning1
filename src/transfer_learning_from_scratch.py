from src.utils.common import read_config
from src.utils.data_mgmt import get_data
from src.utils.model2 import create_model, save_model, save_plot
from src.utils.callbacks import get_callbacks
import argparse
import os
import pandas as pd
import logging
import numpy as np
import time

# return list of labels
def update_even_odd_labels(list_of_labels):
    for idx, label in enumerate(list_of_labels):
        even_condition = label%2==0
        list_of_labels[idx] = np.where(even_condition, 1, 0)
    return list_of_labels

def training(config_path):
    config = read_config(config_path)

    # set the logging details
    logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
    dir1 = config["logs"]["logs_dir"]
    dir2 = config["logs"]["general_logs"]
    general_logs = os.path.join(dir1, dir2) 
    os.makedirs(general_logs, exist_ok=True)
    logging.basicConfig(filename=os.path.join(general_logs, 'training_scratch_logs.log'), level=logging.INFO, format=logging_str, filemode='a')

    validation_datasize = config["params"]["validation_datasize"]
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)

    y_train_bin, y_test_bin, y_valid_bin = update_even_odd_labels([y_train, y_test, y_valid]) 

    LOSS_FUNCTION = config["params"]["loss_function"]
    OPTIMIZER = config["params"]["optimizer"]
    METRICS = config["params"]["metrics"]
    NUM_CLASSES = config["params"]["num_classes"]

    scratch_model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES)

    # Callbacks and Modelcheckpoint
    CALLBACKS_LIST = get_callbacks(config, X_train)  

    # Epochs and validation set
    EPOCHS = config["params"]["epochs"] 
    VALIDATION_SET = (X_valid, y_valid_bin)

    start = time.time()
    history = scratch_model.fit(X_train, y_train_bin, epochs=EPOCHS, validation_data=VALIDATION_SET,
                        callbacks=CALLBACKS_LIST, verbose=2)
    end = time.time()
    total = start - end
    logging.info(f"********model is trained in time: {total}********")

    # Fetching the path for the model and the plot
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_dir = config["artifacts"]["model_dir"]
    plot_dir = config["artifacts"]["plots_dir"]


    # saving the model
    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)
    model_name = config["artifacts"]["scratch_model_name"]
    save_model(scratch_model, model_name, model_dir_path)
    logging.info(f"********model is saved at*********{model_dir_path}")
    logging.info(f"********evaluation metrics *********{scratch_model.evaluate(X_test, y_test_bin)}")

    # saving the path
    plot_dir_path = os.path.join(artifacts_dir, plot_dir)
    os.makedirs(plot_dir_path, exist_ok=True)
    plot_name = config["artifacts"]["plot_name"]
    df = pd.DataFrame(history.history).plot(figsize=(10, 7))
    save_plot(df, plot_name, plot_dir_path)
    logging.info(f"********plot is saved at*********{plot_dir_path}")



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # In the run time itself without editing the code we can pass the arguments from the command line
    args.add_argument("--config", "-c", default="config.yaml")
    parsed_args = args.parse_args()

    training(config_path=parsed_args.config)
    try:
        logging.info("\n************************")
        logging.info(">>>>>>>training has ended <<<<<<<")
    except Exception as e:
        logging.exception(e)
        raise e
