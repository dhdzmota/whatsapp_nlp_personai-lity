from itertools import filterfalse
from unidecode import unidecode


repeated_usual_words = [
    'haha', 'jaja', 'wuwu', 'wowo', 'โคโค', '๐ค๐ค', '๐๐', '๐ณ๐ณ', '๐๐',
    '๐ฅฐ๐ฅฐ', '๐ฅบ๐ฅบ', '๐ฎ๐ฎ', '๐๐', '๐๐', '๐ค๐ค', 'โนโน', '๐ค๐ป๐ค๐ป', '๐คญ๐คญ',
    '๐๐', '๐๐', '๐๐', '๐จ๐จ', '๐ช๐ช', '๐ค๐ค', '๐๐', 'โจโจ',
    '๐ฐ๐ฐ', '๐๐ป๐๐ป', '๐ฌ๐ฌ', '๐๐', '๐ญ๐ญ', '๐ถ๐ถ', '๐๐', '๐ช๐ป๐ช๐ป', '๐ข๐ข',
    '๐คค๐คค', '๐ฑ๐ฑ', 'jsjs', 'jiji', 'hehe', 'jeje',
]


def clean_string(text):
    text = text.lower()
    text = unidecode(text)
    return text


def reduction_of_repeated_strings(string):
    sentence = string.split(' ')
    new_sentence_list = []
    for word in sentence:
        if check_for_repeted_usual_word(word):
            new_word = ''.join(list(unique_everseen(word)))
        else:
            new_word = word
        new_sentence_list.append(new_word)
    return ' '.join(new_sentence_list)


def unique_everseen(iterable, key=None):
    seen = set()
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen.add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen.add(k)
                yield element


def replacements(string):
    string = string.replace('?', ' questionmark ')
    string = string.replace('!', ' exclamationmark ')
    string = string.replace('.', ' dotpunctuation ')
    string = string.replace(',', ' comapunctuation ')
    string = string.replace('):', ' sadface ')
    string = string.replace('(:', ' happyface ')
    string = string.replace('o:', ' surpriseface ')
    string = string.replace('s:', 'worriedface')
    string = string.replace('d:', 'worriedface')
    string = string.replace(':p', 'playfullface')
    return string


def check_for_repeted_usual_word(word):
    global repeated_usual_words
    is_repetead_usual_word_in_word = []
    for w in repeated_usual_words:
        if w in word:
            val = True
        else:
            val = False
        is_repetead_usual_word_in_word.append(val)
    return any(is_repetead_usual_word_in_word)
