from internal.process_input import input_tokenize, match_tokens
from internal.process_files import get_movies

search_limit = 5

def keyword_search(query, index) -> list[dict]:
    query_tokens = input_tokenize(query)

    search_result = set()
    for token in query_tokens:
        results = index.get_documents(token)
        for result in results:
            search_result.add(result)
            if len(search_result) == search_limit:
                return list(search_result)

    '''
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
    '''

    return search_result