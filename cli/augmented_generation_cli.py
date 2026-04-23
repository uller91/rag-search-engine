import argparse
from internal.hybrid_search import SEARCH_LIMIT, HybridSearch, RRF_K
from internal.process_files import get_movies
from internal.gemini import rag

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            movies = get_movies()
            hybrid = HybridSearch(movies)
            query = args.query

            search_result = hybrid.rrf_search(query, RRF_K, SEARCH_LIMIT)
            rag_response = rag(query, search_result)

            print("Search Results:")
            for result in search_result:
                print(f"- {result[1]["document"]["title"]}")
            print("")
            print("RAG Response:")
            print(rag_response)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()