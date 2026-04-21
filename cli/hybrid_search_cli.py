import argparse
from internal.hybrid_search import normalize_command, SEARCH_LIMIT, SEARCH_ALPHA, PRINT_LIMIT, HybridSearch, RRF_K, improve_result_cross_encoder
from internal.process_files import get_movies
from internal.gemini import improve_query, improve_result, evaluate_results

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize the set of scores")
    normalize_parser.add_argument("scores", type=str, nargs='+', help="Scores to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Weighted search movies with BM25 and all-MiniLM-L6-v2 models")
    weighted_search_parser.add_argument("query", type=str, help="Query to search")
    weighted_search_parser.add_argument("-l", "--limit", type=int, nargs='?', default=SEARCH_LIMIT, help="Optional search limit")
    weighted_search_parser.add_argument("-a", "--alpha", type=float, nargs='?', default=SEARCH_ALPHA, help="Optional search alpha parameter")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="RRF search movies with BM25 and all-MiniLM-L6-v2 models")
    rrf_search_parser.add_argument("query", type=str, help="Query to search")
    rrf_search_parser.add_argument("-l", "--limit", type=int, nargs='?', default=SEARCH_LIMIT, help="Optional search limit")
    rrf_search_parser.add_argument("-k", type=int, nargs='?', default=RRF_K, help="Optional RRF k parameter")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rerank = ["individual", "batch", "cross_encoder"]
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=rerank, help="Method to improve RRF search ranking")
    rrf_search_parser.add_argument("--evaluate", action="store_true", help="Evaluate answers with Gemini")

    args = parser.parse_args()

    match args.command:
        case "rrf-search":
            movies = get_movies()
            hybrid = HybridSearch(movies)

            query = args.query
            if args.enhance != None:
                query = improve_query(args.enhance, args.query)
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")

            limit = args.limit
            if args.rerank_method in rerank: 
                limit *= 5

            search_result = hybrid.rrf_search(query, args.k, limit)
            if args.rerank_method in ["individual", "batch"]:
                search_result = improve_result(args.rerank_method, args.query, search_result)
            if args.rerank_method == "cross_encoder":
                search_result = improve_result_cross_encoder(args.query, search_result)
                    
            if args.rerank_method in rerank:
                print(f"Re-ranking top {args.limit} results using {args.rerank_method} method...")
                print(f"Reciprocal Rank Fusion Results for '{args.query}' (k={args.k}):")

                i = 1
                for result in search_result:
                    if i > args.limit:
                        return
                    if "keyword_rank" not in result[1]:
                        result[1]["keyword_rank"] = "NA"
                    if "semantic_rank" not in result[1]:
                        result[1]["semantic_rank"] = "NA"

                    print(f"\n{i}. {result[1]["document"]["title"]}")
                    
                    if args.rerank_method == "cross_encoder":
                        print(f"   Cross Encoder Score: {(result[2]):.4f}")
                    if args.rerank_method == "individual":
                        print(f"   Re-rank Score: {int(result[2])}/10")
                    if args.rerank_method == "batch":
                        print(f"   Re-rank Score: {int(result[2])}")

                    print(f"   RRF Score: {result[1]["rrf_score"]:.4f} ")
                    print(f"   BM25 Rank: {result[1]["keyword_rank"]}, Semantic Rank: {result[1]["semantic_rank"]} ")
                    print(f"   {result[1]["document"]["description"][:PRINT_LIMIT]}...")

                    i += 1
            
            # no rerank
            else:
                i = 1
                for result in search_result:
                    if "keyword_rank" not in result[1]:
                        result[1]["keyword_rank"] = "NA"
                    if "semantic_rank" not in result[1]:
                        result[1]["semantic_rank"] = "NA"

                    print(f"\n{i}. {result[1]["document"]["title"]}")
                    print(f"   RRF Score: {result[1]["rrf_score"]:.4f} ")
                    print(f"   BM25 Rank: {result[1]["keyword_rank"]}, Semantic Rank: {result[1]["semantic_rank"]} ")
                    print(f"   {result[1]["document"]["description"][:PRINT_LIMIT]}...")

                    i += 1
       
            if args.evaluate:
                evaluated_results = evaluate_results(query, search_result[:args.limit])

                i = 1
                for result in evaluated_results:    
                    print(f"\n{i}. {result[1]["document"]["title"]}: {result[2]}/3")

                    i += 1
                
        case "weighted-search":
            movies = get_movies()
            hybrid = HybridSearch(movies)
            search_result = hybrid.weighted_search(args.query, args.alpha, args.limit)

            i = 1
            for result in search_result:
                print(f"\n{i}. {result[1]["document"]["title"]}")
                print(f"   Hybrid Score: {result[1]["hybrid_score"]:.4f} ")
                print(f"   BM25: {result[1]["keyword_score"]:.4f}, Semantic: {result[1]["semantic_score"]:.4f} ")
                print(f"   {result[1]["document"]["description"][:PRINT_LIMIT]}...")

                i += 1
        case "normalize":
            if len(args.scores) != 0:       
                normalized_scores = normalize_command(args.scores)

                for score in normalized_scores:
                    print(f"* {score:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()