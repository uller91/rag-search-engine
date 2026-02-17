from internal.process_input import input_tokenize, match_tokens
from internal.process_files import get_movies

search_limit = 5

def keyword_search(query) -> list[dict]:
    query_tokens = input_tokenize(query)

    movies = get_movies()
    search_result = []
    i = 0
    for film in movies:
        if i >= search_limit:
            break
        title_tokens = input_tokenize(film["title"])
        if match_tokens(query_tokens, title_tokens):
            search_result.append(film)
            i += 1

    return search_result