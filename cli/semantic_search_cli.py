#!/usr/bin/env python3

import argparse
from internal.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, SemanticSearch, SEARCH_LIMIT, ChunkedSemanticSearch, PRINT_LIMIT
from internal.process_files import get_movies
from internal.chunking import chunk_command, semantic_chunk_command, CHUNK_SIZE, MAX_CHUNK_SIZE, OVERLAP

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

    chunk_parser = subparsers.add_parser("chunk", help="Divide the given text in the chunks of custom size (default - 200 words)")
    chunk_parser.add_argument("text", type=str, help="Text to divide")
    chunk_parser.add_argument("--chunk-size", type=int, nargs='?', default=CHUNK_SIZE, help="Optional size of chunks")
    chunk_parser.add_argument("--overlap", type=int, nargs='?', default=OVERLAP, help="Optional chunks overlap")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Divide the given text in the chunks of custom size (default - 4 sentences)")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to divide")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, nargs='?', default=MAX_CHUNK_SIZE, help="Optional size of chunks")
    semantic_chunk_parser.add_argument("--overlap", type=int, nargs='?', default=OVERLAP, help="Optional chunks overlap")

    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Embed the description of moview in chunks with all-MiniLM-L6-v2 model")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search movies with all-MiniLM-L6-v2 model")
    search_chunked_parser.add_argument("query", type=str, help="Query to search")
    search_chunked_parser.add_argument("-l", "--limit", type=int, nargs='?', default=SEARCH_LIMIT, help="Optional search limit")

    args = parser.parse_args()

    match args.command:
        case "search_chunked":
            chunked_semantic = ChunkedSemanticSearch()
            movies = get_movies()
            embeddings = chunked_semantic.load_or_create_chunk_embeddings(movies)
            search_result = chunked_semantic.search_chunked(args.query, args.limit)

            i = 1
            for result in search_result:
                print(f"\n{i}. {result["title"]} {result["id"]} (score: {result["score"]:.4f})")
                print(f"   {result["description"]}...")

                i += 1
        case "embed_chunks":
            chunked_semantic = ChunkedSemanticSearch()
            movies = get_movies()
            embeddings = chunked_semantic.load_or_create_chunk_embeddings(movies)
            print(f"Generated {len(embeddings)} chunked embeddings")
        case "semantic_chunk":
            chunks = semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)

            print(f"Semantically chunking {len(args.text)} characters")
            i = 1
            for chunk in chunks:
                print(f"{i}. {chunk}")
                i += 1
        case "chunk":
            chunks = chunk_command(args.text, args.chunk_size, args.overlap)

            print(f"Chunking {len(args.text)} characters")
            i = 1
            for chunk in chunks:
                print(f"{i}. {chunk}")
                i += 1
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