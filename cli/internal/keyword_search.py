from internal.process_input import input_tokenize, match_tokens
from internal.process_files import get_movies

SEARCH_LIMIT = 5

def keyword_search(query, index) -> list[int]:
    query_tokens = input_tokenize(query)

    search_result = set()
    for token in query_tokens:
        results = index.get_documents(token)
        for result in results:
            search_result.add(result)
            if len(search_result) == SEARCH_LIMIT:
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

def bm25_idf_command(term: str) -> float:
    index = InvertedIndex()
    try:
        index.load() #load an index from cache
    except Exception as e:
        print(e)
        return

    return float(index.get_bm25_idf(term))

def bm25_tf_command(doc_id, term, k1 = BM25_K1, b = BM25_B) -> float:
    index = InvertedIndex()
    try:
        index.load() #load an index from cache
    except Exception as e:
        print(e)
        return

    return float(index.get_bm25_tf(doc_id, term, k1, b))

def bm25search_command(query, limit = SEARCH_LIMIT) -> None:
    index = InvertedIndex()
    try:
        index.load() #load an index from cache
    except Exception as e:
        print(e)
        return

    search_result = index.bm25_search(query, limit)

    i = 1
    for result in search_result:
        print(f"{i}. ({index.docmap[result[0]]["id"]}) {index.docmap[result[0]]["title"]} - Score: {result[1]:.2f}")
        i += 1

    return