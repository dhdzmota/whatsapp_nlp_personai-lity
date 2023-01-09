import os
import pandas as pd

from sklearn.model_selection import train_test_split

from src.data.utils import read_config_file_with_key, read_csv, save_csv
from src.features.build_features import make_real_date_a_date

KEY = 'interim_data'
SUBKEY = 'file2'
SUBKEY_2 = 'sets'
TRAIN_WINDOW_KEY = 'train_window'
PAST_WINDOW_KEY = 'past_window'
FUTURE_WINDOW_KEY = 'future_window'
START_KEY = 'start'
END_KEY = 'end'
RANDOM_STATE_KEY = 'random_state'


def get_between_dates(config_file, key, start, end):
    start_date = read_config_file_with_key(
        file_path=config_file,
        key=key,
        sub_key=start,
    )
    end_date = read_config_file_with_key(
        file_path=config_file,
        key=key,
        sub_key=end,
    )
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    return start_date, end_date


def save_datasets(dataset_dict, path):
    for key in dataset_dict:
        file_name = f'data__{key}.csv'
        file_path = os.path.join(path, file_name)
        save_csv(dataset_dict[key], file_path)

def get_datasets(key_list, path):
    datasets = {}
    for key in key_list:
        file_name = f'data__{key}.csv'
        file_path = os.path.join(path, file_name)
        datasets[key] = read_csv(file_path)
    return datasets


def main():
    general_path = os.path.join(os.path.dirname(__file__), '..', '..')
    data_path = os.path.join(general_path, 'data')
    interim_data_path = os.path.join(data_path, 'interim')
    config_path = os.path.join(general_path, 'config')
    config_file = os.path.join(config_path, 'config.yaml')
    model_config_file = os.path.join(config_path, 'model.yaml')
    file = read_config_file_with_key(
        file_path=config_file,
        key=KEY,
        sub_key=SUBKEY,
    )
    file_path = os.path.join(interim_data_path, file)
    random_state = read_config_file_with_key(
        file_path=model_config_file,
        key=RANDOM_STATE_KEY
    )
    train_start, train_end = get_between_dates(
        config_file=model_config_file,
        key=TRAIN_WINDOW_KEY,
        start=START_KEY,
        end=END_KEY
    )
    past_start, past_end = get_between_dates(
        config_file=model_config_file,
        key=PAST_WINDOW_KEY,
        start=START_KEY,
        end=END_KEY
    )
    future_start, future_end = get_between_dates(
        config_file=model_config_file,
        key=FUTURE_WINDOW_KEY,
        start=START_KEY,
        end=END_KEY
    )
    data = read_csv(file_path)
    data = make_real_date_a_date(data)

    train_window_data = data[data.real_date.between(train_start, train_end)]
    past_window_data = data[data.real_date.between(past_start, past_end)]
    future_window_data = data[data.real_date.between(future_start, future_end)]

    datasets = {}
    datasets['past'] = past_window_data
    datasets['train'], datasets['test'] = train_test_split(
        train_window_data,
        train_size=0.7,
        random_state=random_state
    )
    datasets['test'], datasets['dev'] = train_test_split(
        datasets['test'],
        train_size=0.6,
        random_state=random_state
    )
    datasets['future'] = future_window_data
    save_datasets(datasets, interim_data_path)


if __name__ == '__main__':
    main()
