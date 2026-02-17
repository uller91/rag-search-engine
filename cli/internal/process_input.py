import string
from internal.process_files import get_stop_words
from nltk.stem import PorterStemmer

def input_clean(input) -> str:
    frm = ""
    to = ""
    remove = string.punctuation
    clean_table = str.maketrans(frm, to, remove)
    return (input.lower()).translate(clean_table)

def input_tokenize(input) -> list[str]:
    clean_input = input_clean(input)
    token_input = list(filter(lambda x: x != " ", clean_input.split()))
    no_stop_words_token_input = remove_stop_words(token_input, get_stop_words())
    stemmed_input = reduce_to_stem(no_stop_words_token_input)
    #print(stemmed_input)
    return stemmed_input

def match_tokens(tkns1, tkns2) -> bool:
    for tk1 in tkns1:
        for tkn2 in tkns2:
            if tk1 in tkn2:
                return True
    return False

def remove_stop_words(input, stop_list) -> list[str]:
    return list(filter(lambda x: x not in stop_list, input))

def reduce_to_stem(input) -> list[str]:
    stemmer = PorterStemmer()
    stemmed_input = []
    for token in input:
        stemmed_word = stemmer.stem(token)
        stemmed_input.append(stemmed_word)
    return list(set(stemmed_input))