import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from sklearn.metrics import recall_score, roc_auc_score, accuracy_score

from src.data.utils import read_config_file
from src.models.evaluate_model import predict_from_text, decide_from_text
from src.utils import (
    load_general_model,
    model_predict,
    read_txt_phrases_as_list,
    save_pickle
)


def get_scores_and_bins_for_datasets(all_data_dict, steps, model):
    for key in all_data_dict.keys():
        score = model_predict(model, all_data_dict[key]['x'])
        all_data_dict[key]['score'] = pd.DataFrame(pd.Series(score),
                                                   columns=['score'])
        categories, score_bins = pd.qcut(all_data_dict[key]['score'].score,
                                         q=steps, retbins=True)
        all_data_dict[key]['score']['score_categories'] = categories.to_list()
        all_data_dict[key]['score']['y'] = all_data_dict[key]['y'].to_list()
    return all_data_dict, score_bins

def get_threshold_bounds_for_categs(
    score_df,
    score_bins,
    tolerance=10,
):
    percentage = (100-tolerance)/100
    grouped_score = score_df.groupby('score_categories').agg(
        y=('y', 'mean'),
        count_y=('y', 'count'),
        inverse_y=('y', lambda y: np.mean(1-y)),
    )
    grouped_score['score_bins'] = score_bins[1:]
    categ_0_thresh_score = grouped_score[
        grouped_score.inverse_y > percentage
    ].score_bins.max()
    categ_1_thresh_score = grouped_score[
        grouped_score.y > percentage
    ].score_bins.min()
    categ0_bounds = 0, categ_0_thresh_score
    categ1_bounds = categ_1_thresh_score, 1
    return categ0_bounds, categ1_bounds, grouped_score

def plot_cutpoints_hist_return_error(
    score_df, grouped_score, score_bins, categ0_bounds, categ1_bounds, path
):
    categ_0_lower_bound, categ_0_upper_bound = categ0_bounds
    categ_1_lower_bound, categ_1_upper_bound = categ1_bounds

    categ_0_thresh_score = categ_0_upper_bound
    categ_1_thresh_score = categ_1_lower_bound

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.plot(grouped_score.score_bins, grouped_score.inverse_y, alpha=0.3,
             color='green', label='')
    ax2.scatter(grouped_score.score_bins, grouped_score.inverse_y, alpha=0.7,
                color='green')
    ax2.plot(grouped_score.score_bins, grouped_score.y, alpha=0.3,
             color='purple', label='')
    ax2.scatter(grouped_score.score_bins, grouped_score.y, alpha=0.7,
                color='purple')
    ax2.legend()
    ax1.vlines(categ_0_thresh_score, 0, grouped_score.count_y.max(),
               color='green', linestyle='--')
    ax1.vlines(categ_1_thresh_score, 0, grouped_score.count_y.max(),
               color='purple', linestyle='--')
    sns.histplot(score_df[score_df.y == 0].score, bins=score_bins,
                 color='green', alpha=0.3, ax=ax1)
    sns.histplot(score_df[score_df.y == 1].score, bins=score_bins,
                 color='purple', alpha=0.3, ax=ax1)
    plt.xlabel('Score (resultado del modelo)')
    ax2.set_ylabel('Proporción de categoría')
    title_text = f'Definición de los puntos de corte'
    plt.title(title_text)
    error0 = score_df[score_df.score <= categ_0_thresh_score].y.mean()
    error1 = 1-score_df[score_df.score>=categ_1_thresh_score].y.mean()
    x_position_0 = (categ_0_upper_bound + categ_0_lower_bound) / 2
    x_position_1 = (categ_1_upper_bound + categ_1_lower_bound) / 2
    y_position = (grouped_score.count_y.max()+0) / 2
    ax1.text(
        x_position_0,
        y=y_position,
        s=f'Error = {round(error0*100, 2)}%',
        size=7,
        horizontalalignment='center',
        verticalalignment='center',)
    ax1.text(
        x_position_1,
        y=y_position,
        s=f'Error = {round(error1*100, 2)}%',
        size=7,
        horizontalalignment='center',
        verticalalignment='center',
    )
    plt.savefig(
        f'{path}/whatsapp_definicion_punto_corte.pdf',
        bbox_inches='tight'
    )
    plt.close()
    return error0, error1

def plot_scored_phrases(
    results_df,
    categ0_bounds,
    categ1_bounds,
    error0,
    error1,
    response,
    path
):
    categ_0_lower_bound, categ_0_upper_bound = categ0_bounds
    categ_1_lower_bound, categ_1_upper_bound = categ1_bounds

    categ_0_thresh_score = categ_0_upper_bound
    categ_1_thresh_score = categ_1_lower_bound

    results_df.plot(x='prediction', y='position', kind='scatter', color='k',
                    marker='s', label='frase')
    plt.xlim(0, 1)
    plt.fill_between(x=[categ_0_thresh_score, categ_1_thresh_score], y1=0,
                     y2=results_df.position.max(), alpha=0.3, color='green')
    plt.fill_between(x=[categ_0_thresh_score, categ_1_thresh_score], y1=0,
                     y2=results_df.position.max(), alpha=0.3, color='violet')
    plt.fill_between(x=[categ_0_lower_bound, categ_0_upper_bound], y1=0,
                     y2=results_df.position.max(), alpha=0.3, color='green')
    plt.fill_between(x=[categ_1_lower_bound, categ_1_upper_bound], y1=0,
                     y2=results_df.position.max(), alpha=0.3, color='violet')
    for key in response.keys():
        sample_size = results_df.sample(frac=0.05).shape[0]
        first_stuff = ''.join(key.split(' ')[:2])
        x_position = (categ_0_upper_bound+categ_0_lower_bound) / 2
        error = error0
        if response[key]:
            error = error1
            x_position = (categ_1_upper_bound+categ_1_lower_bound) / 2
        plt.text(x_position, y=results_df.position.max()-sample_size,
                 s=first_stuff, size=7,
                 horizontalalignment='center', verticalalignment='center', )
        plt.text(x_position, y=sample_size,
                 s=f'Error = {round(error * 100, 2)}%', size=7,
                 horizontalalignment='center', verticalalignment='center', )
    x_position = (categ_0_upper_bound+categ_1_lower_bound) / 2
    plt.text(x_position, y=results_df.position.max()-sample_size,
             s='No determinado', size=7,
             horizontalalignment='center', verticalalignment='center', )
    plt.ylim(0, results_df.position.max())
    plt.title('Asignación de resultados')
    plt.xlabel('Score (resultado del modelo)')
    plt.ylabel('Posición (número de frase)')
    plt.legend(loc='center', bbox_to_anchor=(1.1, 0.5))
    plt.savefig(
        f'{path}/whatsapp_asignacion_de_resultados.pdf',
        bbox_inches='tight'
    )
    plt.close()

def score_phrase_list(phrase_list):
    results = pd.DataFrame({'phrase': phrase_list})
    results['prediction'] =  results.phrase.apply(predict_from_text)
    results = results.reset_index().rename(columns={'index':'position'})
    return results


def save_thresholds_and_errors(
    categ_0_thresh_score,
    categ_1_thresh_score,
    error0,
    error1,
    path,
):
    result_dict = {
        'categ_0_thresh_score': categ_0_thresh_score,
        'categ_1_thresh_score': categ_1_thresh_score,
        'error0': error0,
        'error1': error1,
    }
    save_pickle(
        stuff_to_pickle=result_dict,
        save_at=path,
        file_wo_extention='thresholds_and_errors'
    )



def main():
    general_path = os.path.join(os.path.dirname(__file__), '..', '..')
    data_path = os.path.join(general_path, 'data')
    external_data_path = os.path.join(data_path, 'external')
    response_path = os.path.join(external_data_path, 'response_data.json')
    response = read_config_file(response_path)
    phrases_path = os.path.join(external_data_path, 'love_phrases.txt')
    phrases_list = read_txt_phrases_as_list(phrases_path)
    processed_data_path = os.path.join(data_path, 'processed/all_datasets.pkl')
    all_data_dict = pd.read_pickle(processed_data_path)
    models_path = os.path.join(general_path, 'models')
    model = load_general_model('model')
    image_path = os.path.join(general_path, 'reports')

    all_data_dict_with_scores, score_bins = get_scores_and_bins_for_datasets(
        all_data_dict,
        steps=100,
        model=model,
    )
    score_df = all_data_dict_with_scores['future']['score']
    c0_bounds, c1_bounds, grouped_score  = get_threshold_bounds_for_categs(
        score_df,
        score_bins=score_bins,
        tolerance=10
    )
    error0, error1 = plot_cutpoints_hist_return_error(
        score_df, grouped_score,
        score_bins=score_bins,
        categ0_bounds=c0_bounds,
        categ1_bounds=c1_bounds,
        path=image_path,
    )
    result_score_df = score_phrase_list(phrases_list)
    plot_scored_phrases(
        result_score_df,
        categ0_bounds=c0_bounds,
        categ1_bounds=c1_bounds,
        error0=error0,
        error1=error1,
        response=response,
        path=image_path
    )
    categ_0_lower_bound, categ_0_upper_bound = c0_bounds
    categ_1_lower_bound, categ_1_upper_bound = c1_bounds

    categ_0_thresh_score = categ_0_upper_bound
    categ_1_thresh_score = categ_1_lower_bound

    save_thresholds_and_errors(
        categ_0_thresh_score,
        categ_1_thresh_score,
        error0,
        error1,
        path=models_path,
    )
    categ0_text = '\n'.join(
        result_score_df[
            result_score_df.prediction <= categ_0_thresh_score
        ].phrase
    )
    categ1_text = '\n'.join(
        result_score_df[
            result_score_df.prediction >= categ_1_thresh_score
        ].phrase
    )
    print(categ0_text)
    print()
    print(categ1_text)


if __name__ == '__main__':
    main()
