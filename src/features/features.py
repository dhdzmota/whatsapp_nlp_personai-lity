import emoji

from src.features.utils import (
    clean_string,
    reduction_of_repeated_strings,
    replacements,
)


WEEK_DAY_DICT = {
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday'
}


def text_len(text):
    if not text:
        return None
    text_len_int = len(text)
    return text_len_int


def extracted_emojis(text):
    if not text:
        return None
    emoji_list = [emj for emj in str(text) if emj in emoji.UNICODE_EMOJI['en']]
    return ''.join(emoji_list)


def word_count_by_space(text):
    if not text:
        return None
    word_count = len(text.split(' '))
    return word_count


def characters_per_word(text):
    if not text:
        return None
    text_len_int = text_len(text)
    word_counted_by_space = word_count_by_space(text)
    char_per_word = text_len_int / word_counted_by_space
    return char_per_word


def emojis_amount(text):
    if not text:
        return None
    extracted_emojis_str = extracted_emojis(text)
    emojis_amounts = len(extracted_emojis_str)
    return emojis_amounts


def week_day_num(date_time):
    if not date_time:
        return None
    week_day_int = date_time.weekday()
    return week_day_int


def week_day(date_time):
    if not date_time:
        return None
    wd_num = week_day_num(date_time)
    wd = WEEK_DAY_DICT.get(wd_num)
    return wd


def date_hour(date_time):
    if not date_time:
        return None
    hour = date_time.hour
    return hour


def reduced_text(text):
    text = clean_string(text)
    new_string = ''
    prev_string = ''
    for character in text:
        if len(new_string) == 0:
            new_string += character
            prev_string = character
        if character == prev_string:
            continue
        else:
            new_string += character
            prev_string = character
    return new_string


def reduced_text_with_emojis(text):
    emoji_stuff = ' ' + ''.join(extracted_emojis(text))
    text = clean_string(text)
    text = reduced_text(text)
    text += emoji_stuff
    text = reduction_of_repeated_strings(text)
    text = text.strip()
    text = replacements(text)
    return text


def reduced_text_with_emojis_demojized(text):
    text = reduced_text_with_emojis(text)
    text = emoji.demojize(text)
    return text
