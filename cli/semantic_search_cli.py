#!/usr/bin/env python3

import argparse
from internal.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify word processing model")
    
    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify embeddings cache")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed the text with all-MiniLM-L6-v2 model")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    embedquery_parser = subparsers.add_parser("embedquery", help="Embed the query with all-MiniLM-L6-v2 model")
    embedquery_parser.add_argument("query", type=str, help="Query to embed")
    
    args = parser.parse_args()

    match args.command:
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