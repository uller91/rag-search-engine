#!/usr/bin/env python3

import argparse
from internal.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, SemanticSearch, SEARCH_LIMIT
from internal.process_files import get_movies

PRINT_LIMIT = 100

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify word processing model")
    
    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify embeddings cache")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed the text with all-MiniLM-L6-v2 model")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    embedquery_parser = subparsers.add_parser("embedquery", help="Embed the query with all-MiniLM-L6-v2 model")
    embedquery_parser.add_argument("query", type=str, help="Query to embed")
    
    search_parser = subparsers.add_parser("search", help="Search movies with all-MiniLM-L6-v2 model")
    search_parser.add_argument("query", type=str, help="Query to search")
    search_parser.add_argument("-l", "--limit", type=int, nargs='?', default=SEARCH_LIMIT, help="Optional search limit")

    args = parser.parse_args()

    match args.command:
        case "search":
            semantic = SemanticSearch()
            movies = get_movies()
            embeddings = semantic.load_or_create_embeddings(movies)
            search_result = semantic.search(args.query, args.limit)

            i = 1
            for result in search_result:
                print(f"{i}. {result["title"]} (score: {result["score"]:.4f})\n   {result["description"][:PRINT_LIMIT]}...\n")
                i += 1
        case "embedquery":
            embed_query_text(args.query)
        case "verify_embeddings":
            verify_embeddings()
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()