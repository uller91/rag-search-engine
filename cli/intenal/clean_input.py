import string

def clean_input(input) -> str:
    frm = ""
    to = ""
    remove = string.punctuation
    clean_table = str.maketrans(frm, to, remove)
    return (input.lower()).translate(clean_table)