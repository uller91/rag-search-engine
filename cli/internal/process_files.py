import json

movies_json_path = "data/movies.json"
stop_words_path = "data/stopwords.txt"
golden_dataset_path = "data/golden_dataset.json"
#image_file_path = "data/paddington.jpeg"

def get_stop_words() -> list[str] :
    with open(stop_words_path, 'r') as f_stw:
        stop_list = (f_stw.read()).splitlines()
    return stop_list

def get_movies() -> list[dict]:
    with open(movies_json_path, 'r') as f:
        data = json.load(f)
    return data["movies"]

def get_golden_dataset():
    with open(golden_dataset_path, 'r') as f:
        data = json.load(f)
    return data["test_cases"]

def get_image(path):
    try:
        with open(path, 'rb') as i:
            return i.read()
    except FileNotFoundError:
        raise Exception("error when trying to read image!")
        return None
