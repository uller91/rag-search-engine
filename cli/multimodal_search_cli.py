import argparse
from internal.multimodal_search import verify_image_embedding, image_search_command

def main():
    parser = argparse.ArgumentParser(description="Multimodal search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser("verify_image_embedding", help="Verify image embedding for multimodal search"    )
    verify_image_embedding_parser.add_argument("path", type=str, help="Image path")

    image_search_parser = subparsers.add_parser("image_search", help="Search movie by image using multimodal search"    )
    image_search_parser.add_argument("path", type=str, help="Image path")

    args = parser.parse_args()

    match args.command:
        case "image_search":
            image_search_command(args.path)

        case "verify_image_embedding":
            verify_image_embedding(args.path)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()