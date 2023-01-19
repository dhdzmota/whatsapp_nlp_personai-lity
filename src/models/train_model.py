import os
import pandas as pd

from xgboost import XGBClassifier

from src.data.utils import read_config_file_with_key
from src.utils import save_pickle


params = {
    'n_estimators': 100000,
    'max_depth': 4,
    'learning_rate': 0.001,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'colsample_bylevel': 0.8,
    'colsample_bynode': 0.8,
    'random_state': 42,
    'n_jobs': -1,
}

KEY = 'processed_data'
SUBKEY = 'file'


def train_model(train_set, eval_set):
    x_train, y_train = train_set

    model = XGBClassifier(**params)
    model.fit(
        x_train,
        y_train,
        eval_set=[train_set, eval_set],
        early_stopping_rounds=300,
        eval_metric=['aucpr'],
    )
    return model

def main():
    general_path = os.path.join(os.path.dirname(__file__), '..', '..')
    data_path = os.path.join(general_path, 'data')
    processed_data_path = os.path.join(data_path, 'processed')
    models_path = os.path.join(general_path, 'models')
    config_path = os.path.join(general_path, 'config')
    config_file = os.path.join(config_path, 'config.yaml')
    model_config_file = os.path.join(config_path, 'model.yaml')
    processed_data_file_name = read_config_file_with_key(
        file_path=config_file,
        key=KEY,
        sub_key=SUBKEY,
    )
    processed_file = os.path.join(processed_data_path, processed_data_file_name)
    all_datasets = pd.read_pickle(f'{processed_file}.pkl')
    set_train = all_datasets['train']['x'], all_datasets['train']['y']
    set_test = all_datasets['test']['x'], all_datasets['test']['y']
    set_dev = all_datasets['dev']['x'], all_datasets['dev']['y']
    set_past = all_datasets['past']['x'], all_datasets['past']['y']
    set_future = all_datasets['future']['x'], all_datasets['future']['y']
    model = train_model(train_set=set_train, eval_set=set_test)
    save_pickle(
        stuff_to_pickle=model,
        save_at=models_path,
        file_wo_extention='model'
    )


if __name__ == '__main__':
    main()
