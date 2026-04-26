import argparse
import mimetypes
from internal.process_files import get_image
from internal.gemini import describe_image


def main():
    parser = argparse.ArgumentParser(description="Describe Image CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    parser.add_argument("-q", "--query", type=str, help="Query to rewrite based on the image")
    parser.add_argument("-i", "--image", type=str, help="Path to the image")

    args = parser.parse_args()

    #start of the CLI process
    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"
    image = get_image(args.image)

    describe_image(image, mime, args.query.strip())

if __name__ == "__main__":
    main()