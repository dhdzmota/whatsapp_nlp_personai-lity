import numpy as np
import os
import pandas as pd

from src.data.utils import read_config_file_with_key, read_text


KEY = 'files'
SUBKEY_1 = 'file1'
KEY2 = 'interim_data'
SUBKEY_2 = 'file'


def split_last_character(string, split_chars):
    desired_splitted_last_reversed, desired_splitted_front_reversed = \
        (string)[::-1].split(split_chars, 1)
    desired_splitted_front = desired_splitted_front_reversed[::-1]
    desired_splitted_last = desired_splitted_last_reversed[::-1]
    return desired_splitted_front, desired_splitted_last


def initial_process_of_conversation_element(conversation_element):
    initial_conversation_string, date_string = split_last_character(
        string=conversation_element,
        split_chars='\n',
    )
    return initial_conversation_string, date_string


def process_date_string(date_string):
    date, time, am_pm = date_string.split(' ')
    time_h, time_m = split_last_character(time, ':')
    if am_pm.startswith('p') and int(time_h) < 12:
        time_h = str((int(time_h) + 12))
    elif am_pm.startswith('a') and int(time_h) == 12:
        time_h = str(0)
    time = ':'.join([time_h, time_m])
    date_time = ' '.join([date, time])
    return date_time


def process_initial_conversation_string(initial_conversation_string):
    name, conversation_text = initial_conversation_string.split(':', 1)
    name = name.replace('. - ', '')
    conversation_text = conversation_text.strip()
    return name, conversation_text


def add_profile_to_error_info(info):
    info = info.replace('. - ', '. - None: ')
    return info


def parse_date(date):
    day, month, extra = date.split('/')
    day = day.zfill(2)
    month = month.zfill(2)
    return '/'.join([day, month, extra])


def get_conversation_and_error_info(text):
    all_splitted_text = text.split('.\xa0m')
    conversation_info = []
    error_info = []
    # Reduce the function name size
    ipoce = initial_process_of_conversation_element
    pics = process_initial_conversation_string
    for i, conversation_element in enumerate(all_splitted_text):
        try:
            initial_conversation_string, date_string = ipoce(
                conversation_element
            )
            name, conversation_text = pics(initial_conversation_string)
            date_time = process_date_string(date_string)
            conversation_info.append([i, name, conversation_text, date_time,])
        except ValueError:
            error_info.append([i, conversation_element])
    return conversation_info, error_info


def correct_conversation_info_with_error_info(conversation_info, error_info):
    ipoce = initial_process_of_conversation_element
    pics = process_initial_conversation_string
    conversation_info_columns = ['position', 'user', 'text', 'date']
    error_info_columns = ['position', 'info']
    text_data = pd.DataFrame(
        conversation_info,
        columns=conversation_info_columns
    )
    error_data = pd.DataFrame(
        error_info,
        columns=error_info_columns
    )
    # Correct last data
    last_error = error_data.iloc[-1]
    last_position = last_error.position
    last_user, last_text = pics(last_error['info'])
    last_date = np.nan
    error_last = pd.Series(
        [last_position, last_user, last_text, last_date],
        index=conversation_info_columns
    )
    text_data = text_data.append(error_last, ignore_index=True)

    # Correct intermediate stuff
    non_desired_values = [0, int(last_position)]
    intermediate_stuff_df = error_data.loc[
        ~error_data.position.isin(non_desired_values)
    ]
    intermediate_info = []
    for row in range(intermediate_stuff_df.shape[0]):
        row_df = intermediate_stuff_df.iloc[row-1]
        initial_conversation_string, date_string = ipoce(
            add_profile_to_error_info(row_df['info'])
        )
        name, conversation_text = process_initial_conversation_string(
            initial_conversation_string)
        date_time = process_date_string(date_string)
        position = row_df.position
        intermediate_info.append(
            [position, name, conversation_text, date_time, ])
    intermediate_data = pd.DataFrame(
        intermediate_info,
        columns=conversation_info_columns
    )
    semi_final_data = pd.concat(
        [text_data, intermediate_data]
    )
    semi_final_data.sort_values('position', inplace=True)
    semi_final_data.reset_index(drop=True, inplace=True)
    first_date = process_date_string(
        error_data[error_data.position == 0].iloc[0]['info'])
    all_dates = semi_final_data.date.dropna().to_list()
    semi_final_data['real_date'] = [first_date] + all_dates
    semi_final_data['real_date'] = semi_final_data.real_date.apply(parse_date)
    semi_final_data.real_date = pd.to_datetime(
        semi_final_data.real_date,
        format="%d/%m/%Y %H:%M"
    )
    final_data = semi_final_data[['user', 'text', 'real_date']]
    final_data.reset_index(drop=True, inplace=True)
    return final_data


def main():
    general_path = os.path.join(os.path.dirname(__file__), '..', '..')
    data_path = os.path.join(general_path, 'data')
    raw_data_path = os.path.join(data_path, 'raw')
    interim_data_path = os.path.join(data_path, 'interim')

    config_filepath = os.path.join(general_path, 'config/config.yaml')
    file = read_config_file_with_key(
        file_path=config_filepath,
        key=KEY,
        sub_key=SUBKEY_1,
    )
    save_file = read_config_file_with_key(
        file_path=config_filepath,
        key=KEY2,
        sub_key=SUBKEY_2,
    )
    raw_file_path = os.path.join(raw_data_path, file)
    data_save_path = os.path.join(interim_data_path, save_file)

    text = read_text(raw_file_path)
    conversation_info, error_info = get_conversation_and_error_info(text)
    data = correct_conversation_info_with_error_info(
        conversation_info=conversation_info,
        error_info=error_info,
    )
    data.to_csv(data_save_path)
    print(f'Data is now saved into {data_save_path}')


if __name__ == '__main__':
    main()
