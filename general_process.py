import os

dir = os.path.join(os.path.dirname(__file__))

command_chains = [
    f'pip install -e {dir}.',
    f'pip install -r {dir}requirements.txt',
    f'python {dir}src/data/download_emojis.py',
    f'python {dir}src/data/process_data.py',
    f'python {dir}src/features/build_features.py',
    f'python {dir}src/features/split.py',
    f'python {dir}src/features/encode_text_information.py',
    f'python {dir}src/models/train_model.py',
    f'python {dir}src/visualization/visualize_and_get_parameters.py',
]

def run_pipeline():
    for command in command_chains:
        print(f'Running the following command: "{command}".')
        os.system(command)
        print('Done running command.')

if __name__ == '__main__':
    run_pipeline()
