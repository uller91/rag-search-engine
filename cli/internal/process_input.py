import string

def input_clean(input) -> str:
    frm = ""
    to = ""
    remove = string.punctuation
    clean_table = str.maketrans(frm, to, remove)
    return (input.lower()).translate(clean_table)

def input_tokenize(input) -> list[str]:
    clean_input = input_clean(input)
    return list(filter(lambda x: x != " ", clean_input.split()))

def match_tokens(tkns1, tkns2) -> bool:
    for tk1 in tkns1:
        for tkn2 in tkns2:
            if tk1 in tkn2:
                return True
    return False

def remove_stop_words(input, stop_list) -> list[str]:
    return list(filter(lambda x: x not in stop_list, input))
