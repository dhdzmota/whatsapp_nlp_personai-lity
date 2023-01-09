import datetime
import pandas as pd

from src.utils import (
    load_general_model,
    save_pickle,
    model_predict,
    invert_dict,
)

from src.features.features import (
    text_len,
    word_count_by_space,
    characters_per_word,
    emojis_amount,
    week_day_num,
    date_hour,
    reduced_text_with_emojis_demojized,
)


def predict_from_text(text):
    vectorizer = load_general_model('vectorizer')
    usable_columns = load_general_model('usable_columns')
    pca = load_general_model('pca')
    model = load_general_model('model')

    now_datetime = datetime.datetime.now()

    processed_text = reduced_text_with_emojis_demojized(text)
    processed_series_text = pd.Series(processed_text)
    transformed_text_data_eval = vectorizer.transform(processed_series_text)
    transformed_text_data_eval_df = pd.DataFrame(
        transformed_text_data_eval.toarray(),
        columns=vectorizer.get_feature_names_out()
    )
    reduced_transformed_text_data_df_eval = transformed_text_data_eval_df[
        usable_columns
    ]
    pca_reduced_transformed_text_data_eval = pca.transform(
        reduced_transformed_text_data_df_eval
    )
    pca_reduced_transformed_text_data_df_eval =pd.DataFrame(
        pca_reduced_transformed_text_data_eval,
        columns=pca.get_feature_names_out()
    )
    final_df = pca_reduced_transformed_text_data_df_eval
    final_df['week_day_num'] = week_day_num(now_datetime)
    final_df['date_hour'] = date_hour(now_datetime)
    final_df['emojis_amount'] = emojis_amount(text)
    final_df['text_len'] = text_len(text)
    final_df['word_count_by_space'] = word_count_by_space(text)
    final_df['characters_per_word'] = characters_per_word(text)
    response = model_predict(model,final_df)[0]
    return  response

def decide_from_text(text, decision0=0, decision1=1, decision_dict=None):
    decision_dict = invert_dict(decision_dict)
    response = predict_from_text(text)
    if response < decision0:
        return decision_dict.get(0)
    if response > decision1:
        return decision_dict.get(1)
    return str(None)



def main():
    # text = 'Te amo mucho mi vida ❤ y si ya vas a dormir! Buenas noches ✨'
    text = input('Write a text: ')
    print(predict_from_text(text))

if __name__ == '__main__':
    main()



