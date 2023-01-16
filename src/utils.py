import os
import pickle

basedir = os.path.join(os.path.dirname(__file__), '..')

def read_txt_phrases_as_list(file_with_path):
    with open(file_with_path, 'r') as f:
        text = f.read()
    text_list = text.split('\n')
    return text_list


def save_pickle(stuff_to_pickle, save_at, file_wo_extention):
    protocol = pickle.HIGHEST_PROTOCOL
    with open(f'{save_at}/{file_wo_extention}.pkl', 'wb') as file:
        pickle.dump(stuff_to_pickle, file, protocol=protocol)

def load_model():
    model_path = os.path.abspath(os.path.join(basedir,'models/model.pkl'))
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def load_general_model(file_wo_extention):
    model_path = os.path.abspath(os.path.join(
        basedir, f'models/{file_wo_extention}.pkl')
    )
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def model_predict(model, df):
    prediction = model.predict_proba(df)[:, 1]
    return prediction

def invert_dict(dictionary):
    yranoitcid = {}
    for key, val in dictionary.items():
        yranoitcid[val] = key
    return yranoitcid