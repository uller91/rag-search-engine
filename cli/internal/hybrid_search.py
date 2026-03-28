import os

from internal.inverted_index import InvertedIndex
from internal.semantic_search import ChunkedSemanticSearch

CACHE_DIR = "cache"
SEARCH_LIMIT = 5
SEARCH_ALPHA = 0.5
PRINT_LIMIT = 100

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.index = InvertedIndex()
        if not os.path.exists(self.index.index_path):
            self.index.build()
            self.index.save()

    def _bm25_search(self, query, limit):
        self.index.load()
        return self.index.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        results_keyword = self._bm25_search(query, limit*500)
        results_semantic = self.semantic_search.search_chunked(query, limit*500)

        scores_keyword = []
        for result in results_keyword:
            scores_keyword.append(result[1])
        scores_keyword_normalized = normalize_command(scores_keyword)
        results_keyword_normalized = []
        for i in range(len(results_keyword)):
            results_keyword_normalized.append([results_keyword[i][0], scores_keyword_normalized[i]])

        scores_semantic = []
        for result in results_semantic:
            scores_semantic.append(result["score"])
        scores_semantic_normalized = normalize_command(scores_semantic)
        for i in range(len(results_semantic)):
            results_semantic[i]["score"] = scores_semantic_normalized[i]

        results = {}
        for result in results_keyword_normalized:
            results[result[0]] = {"document": self.index.docmap[result[0]], "keyword_score": result[1], "semantic_score": 0}
        for result in results_semantic:
            if result["id"] not in results.keys():
                results[result["id"]] = {"document": self.index.docmap[result["id"]], "keyword_score": 0, "semantic_score": result["score"]}
            else:
                results[result["id"]]["semantic_score"] = result["score"]

        for id in results.keys():
            results[id]["hybrid_score"] = results[id]["keyword_score"] * alpha + results[id]["semantic_score"] * (1 - alpha)


        results_sorted = sorted(results.items(), key = lambda x: x[1]["hybrid_score"], reverse=True)
        return_result = []
        j = 0
        for result in results_sorted:
            return_result.append(result)
            j += 1
            if j >= limit:
                break

        return return_result

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

def normalize_command(scores):
    min_score = float(min(scores))
    max_score = float(max(scores))

    normalized = []
    if min_score == max_score:
        for i in range(len(scores)):
            normalized.append(1.0)
    else:
        for i in range(len(scores)):
            normalized.append((float(scores[i])-min_score)/(max_score-min_score))

    return normalized