import json

movies_json_path = "data/movies.json"
stop_words_path = "data/stopwords.txt"

def get_stop_words() -> list[str] :
    with open(stop_words_path, 'r') as f_stw:
        stop_list = (f_stw.read()).splitlines()
    return stop_list

def get_movies() -> list[dict]:
    with open(movies_json_path, 'r') as f:
        data = json.load(f)
    return data["movies"]