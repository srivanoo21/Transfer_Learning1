from src.utils.common import read_config
from src.utils.data_mgmt import get_data
from src.utils.model import create_model, save_model, save_plot
from src.utils.callbacks import get_callbacks
import argparse
import os
import pandas as pd

def training(config_path):
    config = read_config(config_path)

    validation_datasize = config["params"]["validation_datasize"]
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)

    LOSS_FUNCTION = config["params"]["loss_function"]
    OPTIMIZER = config["params"]["optimizer"]
    METRICS = config["params"]["metrics"]
    NUM_CLASSES = config["params"]["num_classes"]

    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES)

    # Callbacks and Modelcheckpoint
    CALLBACKS_LIST = get_callbacks(config, X_train)  

    # Epochs and validation set
    EPOCHS = config["params"]["epochs"] 
    VALIDATION_SET = (X_valid, y_valid)

    history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION_SET,
                        callbacks=CALLBACKS_LIST)

    # Fetching the path for the model and the plot
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_dir = config["artifacts"]["model_dir"]
    plot_dir = config["artifacts"]["plots_dir"]


    # saving the model
    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)
    model_name = config["artifacts"]["model_name"]
    save_model(model, model_name, model_dir_path)

    # saving the path
    plot_dir_path = os.path.join(artifacts_dir, plot_dir)
    os.makedirs(plot_dir_path, exist_ok=True)
    plot_name = config["artifacts"]["plot_name"]
    df = pd.DataFrame(history.history).plot(figsize=(10, 7))
    save_plot(df, plot_name, plot_dir_path)

    # saving the logs




if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # In the run time itself without editing the code we can pass the arguments from the command line
    args.add_argument("--config", "-c", default="config.yaml")
    parsed_args = args.parse_args()

    training(config_path=parsed_args.config)