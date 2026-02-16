import json
from intenal.clean_input import clean_input

search_limit = 5

def json_read(path, query) -> list[dict]:
    with open(path, 'r') as f:
        data = json.load(f)

    search_result = []
    i = 0
    for film in data["movies"]:
        if clean_input(query) in clean_input((film["title"])) and i < search_limit: #case sensitivity
            search_result.append(film)
            i += 1

    return search_result