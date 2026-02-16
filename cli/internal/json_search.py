import json
from internal.process_input import input_tokenize
from internal.check_tokens import check_tokens

search_limit = 5

def json_read(path, query) -> list[dict]:
    with open(path, 'r') as f:
        data = json.load(f)

    query_tokens = input_tokenize(query)

    search_result = []
    i = 0
    for film in data["movies"]:
        if i >= search_limit:
            break
        title_tokens = input_tokenize(film["title"])
        if check_tokens(query_tokens, title_tokens):
            search_result.append(film)
            i += 1

    return search_result