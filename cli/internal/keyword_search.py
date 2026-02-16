import json
from internal.process_input import input_tokenize, match_tokens, remove_stop_words

search_limit = 5
stop_words_path = "data/stopwords.txt"

def keyword_search(path, query) -> list[dict]:
    with open(path, 'r') as f:
        data = json.load(f)

    with open(stop_words_path, 'r') as f_stw:
        stop_list = (f_stw.read()).splitlines()

    query_tokens = remove_stop_words(input_tokenize(query), stop_list)

    search_result = []
    i = 0
    for film in data["movies"]:
        if i >= search_limit:
            break
        title_tokens = remove_stop_words(input_tokenize(film["title"]), stop_list)
        if match_tokens(query_tokens, title_tokens):
            search_result.append(film)
            i += 1

    return search_result