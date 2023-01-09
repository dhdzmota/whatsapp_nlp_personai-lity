import json
import os
import pandas as pd

from src.data.utils import read_config_file_with_key, read_csv, save_csv
from src.features.features import (
    extracted_emojis,
    text_len,
    word_count_by_space,
    characters_per_word,
    emojis_amount,
    week_day_num,
    week_day,
    date_hour,
    reduced_text,
    reduced_text_with_emojis,
    reduced_text_with_emojis_demojized,
)


KEY = 'interim_data'
KEY2 = 'external_data'
SUBKEY = 'file'
SUBKEY2 = 'file2'
UNNECESSARY_MESSAGES = [
    '<Multimedia omitido>',
    'Esperando este mensaje',
]

def save_response_information(categ, path):
    json_object = json.dumps(categ, indent=4)
    with open(path,'w') as out_file:
        out_file.write(json_object)


def remove_unnecessary_messages_and_users(data):
    data_cleaned = data.copy()
    for message in UNNECESSARY_MESSAGES:
        data_cleaned = data_cleaned[data.text != message]
    data_cleaned = data_cleaned[data_cleaned.user != 'None']
    return data_cleaned


def make_real_date_a_date(data):
    data['real_date'] = pd.to_datetime(data['real_date'])
    return data


def build_features(data):
    print('Computing features...')
    data['extracted_emojis'] = data['text'].apply(extracted_emojis)
    data['week_day_num'] = data['real_date'].apply(week_day_num)
    data['week_day'] = data['real_date'].apply(week_day)
    data['date_hour'] = data['real_date'].apply(date_hour)
    data['emojis_amount'] = data['text'].apply(emojis_amount)
    data['text_len'] = data['text'].apply(text_len)
    data['word_count_by_space'] = data['text'].apply(word_count_by_space)
    data['characters_per_word'] = data['text'].apply(characters_per_word)
    data['reduced_text'] = data['text'].apply(reduced_text)
    data['reduced_text_with_emojis'] = data['text'].apply(
        reduced_text_with_emojis
    )
    data['reduced_text_with_emojis_demojized'] = data['text'].apply(
        reduced_text_with_emojis_demojized
    )
    # Category is a different feature since it dependes on the actual
    # categories, therefore knowledge on all the categories must be known to
    # compute.
    print('Computing response variable.')
    indexes = data.user.value_counts().index.to_list()
    categ = {name: i for i, name in enumerate(indexes)}
    data['categ_y'] = data['user'].map(categ)
    print('Done with features and response variable.')
    return data, categ


def main():
    general_path = os.path.join(os.path.dirname(__file__), '..', '..')
    data_path = os.path.join(general_path, 'data')
    interim_data_path = os.path.join(data_path, 'interim')
    external_data_path = os.path.join(data_path, 'external')
    config_filepath = os.path.join(general_path, 'config/config.yaml')
    file = read_config_file_with_key(
        file_path=config_filepath,
        key=KEY,
        sub_key=SUBKEY,
    )
    save_file = read_config_file_with_key(
        file_path=config_filepath,
        key=KEY,
        sub_key=SUBKEY2,
    )
    save_response_file = read_config_file_with_key(
        file_path=config_filepath,
        key=KEY2,
        sub_key=SUBKEY,
    )
    file_path = os.path.join(interim_data_path, file)
    save_file_path = os.path.join(interim_data_path, save_file)
    save_response_path = os.path.join(external_data_path, save_response_file)
    data = read_csv(file_path)
    data = make_real_date_a_date(data)
    data = remove_unnecessary_messages_and_users(data)
    data, categ = build_features(data)
    save_response_information(categ=categ, path=save_response_path)
    save_csv(data, save_file_path)


if __name__ == '__main__':
    main()
