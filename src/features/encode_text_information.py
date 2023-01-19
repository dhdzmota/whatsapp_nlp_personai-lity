import os
import pandas as pd
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA

from src.data.utils import read_config_file_with_key, read_csv, save_csv
from src.features.build_features import make_real_date_a_date
from src.features.split import get_datasets
from src.utils import load_general_model, save_pickle


KEY = 'interim_data'
KEY2 = 'processed_data'
SUBKEY = 'sets'
SUBKEY2 = 'file'
RANDOM_STATE_KEY = 'random_state'
COUNT_VECTORIZER_KEY = 'count_vectorizer'
STOP_WORDS_SUBKEY = 'stop_words_list'
PCA_KEY='pca'
PCA_COMPONENTS_SUBKEY = 'principal_components'


def get_usable_columns_by_sum(df, thresh=3):
    col_sum = df.sum()
    usable_columns_filter = col_sum[col_sum > thresh]
    usable_columns = usable_columns_filter.index
    return usable_columns


def fit_text_transformers(
    train_data,
    past_data=None,
    stop_words_list=None,
    pca_components=None,
    save_at=None
):
    data = train_data.copy()
    if not (past_data is None):
        data = pd.concat([train_data, past_data])
    data = data[data.reduced_text_with_emojis_demojized.notna()]
    text_data = data.reduced_text_with_emojis_demojized
    print('Generation of vectorizer for text...')
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        token_pattern=r'\b\w+\b',
        stop_words=stop_words_list,
    )
    print('Fitting vectorizer for text...')
    vectorizer.fit(text_data)
    print('Transform text with vectorizer...')
    transformed_text_data = vectorizer.transform(text_data)
    transformed_text_data_df = pd.DataFrame(
        transformed_text_data.toarray(),
        columns=vectorizer.get_feature_names_out(),
    )
    print('Getting only the usable columns of the transformed '
          'text with vectorizer...')
    usable_columns = get_usable_columns_by_sum(
        df=transformed_text_data_df,
        thresh=3
    )
    reduced_transformed_text_data_df = transformed_text_data_df[
        usable_columns
    ]
    print('Generating PCA...')
    pca = PCA(n_components=pca_components)
    print('Fitting PCA for text...')
    pca.fit(reduced_transformed_text_data_df)
    protocol = pickle.HIGHEST_PROTOCOL
    print(f'Saving vectorizer, usable_columns and pca in: {save_at}')
    with open(f'{save_at}/vectorizer.pkl', 'wb') as model_vec:
        pickle.dump(vectorizer, model_vec, protocol=protocol)
    with open(f'{save_at}/usable_columns.pkl', 'wb') as usable_cols:
        pickle.dump(usable_columns, usable_cols, protocol=protocol)
    with open(f'{save_at}/pca.pkl', 'wb') as model_pca:
        pickle.dump(pca, model_pca, protocol=protocol)
    print('Done with generation of encoders...')


def transform_data(data):
    print('Transforming data...')
    data = data[data.reduced_text_with_emojis_demojized.notna()]
    vectorizer = load_general_model('vectorizer')
    usable_columns = load_general_model('usable_columns')
    pca = load_general_model('pca')

    transformed_text_data = vectorizer.transform(
        data.reduced_text_with_emojis_demojized
    )
    transformed_text_data_df = pd.DataFrame(
        transformed_text_data.toarray(),
        columns=vectorizer.get_feature_names_out()
    )
    transformed_text_data_df = transformed_text_data_df[usable_columns]
    reduced_transformed_text_data_df = transformed_text_data_df[usable_columns]
    pca_reduced_transformed_text_data = pca.transform(
        reduced_transformed_text_data_df
    )
    pca_transformed_text_data_df = pd.DataFrame(
        pca_reduced_transformed_text_data,
        columns=pca.get_feature_names_out()
    )
    pca_transformed_text_data_df = pca_transformed_text_data_df.set_index(
        data.index
    )
    x_data = pca_transformed_text_data_df
    x_data['week_day_num'] = data['week_day_num']
    x_data['date_hour'] = data['date_hour']
    x_data['emojis_amount'] = data['emojis_amount']
    x_data['text_len'] = data['text_len']
    x_data['word_count_by_space'] = data['word_count_by_space']
    x_data['characters_per_word'] = data['characters_per_word']
    y_data = data['categ_y']
    print('Done.')
    return x_data, y_data


def main():
    general_path = os.path.join(os.path.dirname(__file__), '..', '..')
    data_path = os.path.join(general_path, 'data')
    interim_data_path = os.path.join(data_path, 'interim')
    processed_data_path = os.path.join(data_path, 'processed')
    models_path = os.path.join(general_path, 'models')
    config_path = os.path.join(general_path, 'config')
    config_file = os.path.join(config_path, 'config.yaml')
    model_config_file = os.path.join(config_path, 'model.yaml')
    random_state = read_config_file_with_key(
        file_path=model_config_file,
        key=RANDOM_STATE_KEY
    )
    datasets_list = read_config_file_with_key(
        file_path=config_file,
        key=KEY,
        sub_key=SUBKEY,
    )

    processed_data_file_name = read_config_file_with_key(
        file_path=config_file,
        key=KEY2,
        sub_key=SUBKEY2,
    )

    stop_words_list = read_config_file_with_key(
        file_path=model_config_file,
        key=COUNT_VECTORIZER_KEY,
        sub_key=STOP_WORDS_SUBKEY
    )
    pca_components = read_config_file_with_key(
        file_path=model_config_file,
        key=PCA_KEY,
        sub_key=PCA_COMPONENTS_SUBKEY,
    )
    datasets = get_datasets(key_list=datasets_list, path=interim_data_path)
    fit_text_transformers(
        train_data=datasets['train'],
        past_data=datasets['past'].sample(frac=0.5, random_state=random_state),
        stop_words_list=stop_words_list,
        pca_components=pca_components,
        save_at=models_path
    )
    model_datasets = {}
    for key in datasets.keys():
        model_datasets[key] = {}
        model_datasets[key]['x'], model_datasets[key]['y']  = transform_data(
            datasets[key]
        )
    save_pickle(
        stuff_to_pickle=model_datasets,
        save_at=processed_data_path,
        file_wo_extention=processed_data_file_name,
    )


if __name__ == '__main__':
    main()
