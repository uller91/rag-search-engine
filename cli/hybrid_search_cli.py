import argparse
from internal.hybrid_search import normalize_command, SEARCH_LIMIT, SEARCH_ALPHA, PRINT_LIMIT, HybridSearch
from internal.process_files import get_movies

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize the set of scores")
    normalize_parser.add_argument("scores", type=str, nargs='+', help="Scores to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Weighted search movies with BM25 and all-MiniLM-L6-v2 models")
    weighted_search_parser.add_argument("query", type=str, help="Query to search")
    weighted_search_parser.add_argument("-l", "--limit", type=int, nargs='?', default=SEARCH_LIMIT, help="Optional search limit")
    weighted_search_parser.add_argument("-a", "--alpha", type=float, nargs='?', default=SEARCH_ALPHA, help="Optional search alpha parameter")

    args = parser.parse_args()

    match args.command:
        case "weighted-search":
            movies = get_movies()
            hybrid = HybridSearch(movies)
            search_result = hybrid.weighted_search(args.query, args.alpha, args.limit)

            i = 1
            for result in search_result:
                print(f"\n{i}. {result["document"]["title"]}")
                print(f"   Hybrid Score: {result["hybrid_score"]:.4f} ")
                print(f"   BM25: {result["keyword_score"]:.4f}, Semantic: {result["semantic_score"]:.4f} ")
                print(f"   {result["document"]["description"][:PRINT_LIMIT]}...")

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