import re

CHUNK_SIZE = 200
MAX_CHUNK_SIZE = 4
OVERLAP = 0

def semantic_chunk_command(text, size, overlap):
    if overlap < 0:
        overlap = 0

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    parts = []
    i = 0
    while len(sentences) > 0:
        if len(sentences) <= size:
            parts = sentences
            sentences = []
        else:
            parts = sentences[:size]
            sentences = sentences[size-overlap:]

        chunk = " ".join(parts)
        chunks.append(chunk)

    return chunks

def chunk_command(text, size, overlap):
    if overlap < 0:
        overlap = 0

    words = text.split(" ")
    chunks = []
    parts = []
    while len(words) > 0:
        if len(words) <= size:
            parts = words
            words = []
        else:
            parts = words[:size]
            words = words[size-overlap:]

        chunk = " ".join(parts)
        chunks.append(chunk)

    return chunks