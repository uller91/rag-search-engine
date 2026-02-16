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