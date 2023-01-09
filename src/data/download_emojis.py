import base64
import os
import requests

from bs4 import BeautifulSoup

from src.data.utils import read_config_file_with_key


KEY = 'links'
SUBKEY_1 = 'emoji_url'
SUBKEY_2 = 'emoji_modifiers_url'


def download_emoji(link, filepath):
    """
    This function downloads emojis from the desired link (see config) in the
    desired filepath. In this case, the emojis downloaded are images so that
    they can be easily included in a matplotlib plot.

    :param link: str
        Desired url as string.
    :param filepath:
        Desired path where one wants to save all the files.
    :return:
        None
    """
    response = requests.get(link)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table')
    emojilist = []
    table_tr = table.findAll(['tr'])[3:]
    for row in table_tr:
        code = row.find_all('td', attrs={"class": "code"})
        imag = row.find_all('td', attrs={"class": "andr alt"})
        if code and imag:
            emojilist.append((code[0].text, imag[0].img['src']))
    prefix_len = len("data:image/png;base64,")
    for code, data in emojilist:
        code = code[2:]
        code = code.replace(" U+", "_")
        with open(f"{filepath}/{code}.png".lower(), "wb") as fh:
            fh.write(base64.decodestring(bytes(data[prefix_len:], 'utf-8')))
        print(f'Downloading {code}.png')
    return None


def main():
    general_path = os.path.join(os.path.dirname(__file__), '..', '..')
    emoji_path = os.path.join(general_path, 'data/external/emoji')
    config_filepath = os.path.join(general_path, 'config/config.yaml')
    emoji_url = read_config_file_with_key(
        file_path=config_filepath,
        key=KEY,
        sub_key=SUBKEY_1,
    )
    emoji_modifiers_url = read_config_file_with_key(
        file_path=config_filepath,
        key=KEY,
        sub_key=SUBKEY_2,
    )
    if not os.path.exists(emoji_path):
        print(f'Creating path: {emoji_path}')
        os.mkdir(emoji_path)

    if not os.listdir(emoji_path):
        print(f'Downloading emoji files in path:{emoji_path}')
        download_emoji(emoji_url, filepath=emoji_path)
        download_emoji(emoji_modifiers_url, filepath=emoji_path)
        print('Done with download.')
    else:
        print(f'Nothing to download, files are already at: {emoji_path}')


if __name__ == '__main__':
    main()
