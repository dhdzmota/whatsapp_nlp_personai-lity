import gradio as gr
import os

from src.data.utils import read_config_file
from src.models.evaluate_model import predict_from_text, decide_from_text
from src.utils import (
    load_general_model,
)

def process():
    general_path = os.path.join(os.path.dirname(__file__), '..')
    data_path = os.path.join(general_path, 'data')
    external_data_path = os.path.join(data_path, 'external')
    response_path = os.path.join(external_data_path, 'response_data.json')

    response = read_config_file(response_path)
    thresholds_and_errors = load_general_model('thresholds_and_errors')

    func = lambda Escribe: ''.join(decide_from_text(
        Escribe,
        decision0=thresholds_and_errors['categ_0_thresh_score'],
        decision1=thresholds_and_errors['categ_1_thresh_score'],
        decision_dict=response,
    ).split(' ')[:2])

    demo_description = 'PersonAi-lity busca entender el comportamiento' \
                       ' de cada perfil estudiando la conversación de ' \
                       'whatsapp de dos usuarios.'
    demo_article = 'Gracias por usar PersonAi-lity.'

    demo = gr.Interface(
        title='PersonAI-lity',
        description=demo_description,
        article=demo_article,
        fn=func,
        thumbnail='../whatsapp_image_picture.jpg',
        inputs=gr.inputs.Textbox(
            lines=3,
            placeholder='Escribe una frase aquí',
            label='Frase:'
        ),
        outputs=gr.outputs.Textbox(label='Decisión:'),
    )
    demo.launch()

if __name__ == '__main__':
    process()

