import yaml
import pandas as pd


def read_config_file(file_path):
    """
    This function helps o read a yaml file from a specific path.

    :param file_path: str
        string that indicates the path where the file is located, and with the
        name of the file.
    :return: yaml_file dict
        Returns a dictionary with the yaml file information.

    """
    print(f'Reading yaml file at: {file_path}')
    with open(file_path, 'r') as file:
        yaml_file = yaml.safe_load(file)
    return yaml_file


def read_config_file_with_key(file_path, key, sub_key=None):
    """ This function gets a specific key information from a
    config file given the limitation that the yaml file must have at most 2
    levels of information, that is a key and a sub_key. More information wont
    be processed.

    :param file_path: str
        string that indicates the path where the file is located, and with the
        name of the file.
    :param key: str
        string that indicates the desired key to be explored
    :param sub_key: str
        string that indicates the desired sub-key to be explored

    :return: key_info str
        information of the desired key or sub-key in the config file.
    """
    print(f'Reading key: {key}')
    file = read_config_file(file_path)
    key_info = file[key]
    if sub_key:
        print(f'Reading sub_key: {sub_key}')
        key_info = key_info[sub_key]
    return key_info


def read_text(file_path):
    """
    This function reads a txt file.

    :param file_path: str
        desied path of the file.
    :return: str
        string contained in the desired file.

    """
    with open(file_path, 'r') as file:
        text = file.read()
    return text


def read_csv(file_path):
    df = pd.read_csv(file_path, index_col=0)
    return df


def save_csv(df, file_path):
    df.to_csv(file_path)
