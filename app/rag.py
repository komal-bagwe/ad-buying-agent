"""
RAG layer - reads local policy documents, embeds them via Cohere, and
lets the agent search for relevant context before making a bidding
decision.

Source documents live in data/docs/ as plain .txt files, optionally
starting with a SOURCE_URL / SOURCE_TITLE header followed by '---'
(see data/docs/*.txt for examples of both formats).

Run this file directly once to load documents into Qdrant:
    python3 -m app.rag
Then call search_context(query) from anywhere else to retrieve
relevant chunks.
"""

import os
import re
import uuid
from pathlib import Path
from dotenv import load_dotenv
import cohere
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

DOCS_DIR = Path(__file__).parent.parent / "data" 
COHERE_MODEL = "embed-english-v3.0"
EMBEDDING_DIM = 1024  # embed-english-v3.0's output vector size
COLLECTION_NAME = "ad_agent_knowledge"

cohere_client = cohere.Client(api_key=os.environ["COHERE_API_KEY"])
qdrant_client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
)


def _load_and_chunk_documents() -> list[dict]:
    """
    Read every .txt file in data/docs/, split each into paragraph
    chunks, and return them as {text, source_url} dicts.

    Files may optionally start with a SOURCE_URL: ... header followed
    by '---' - this is used for attribution if present. Plain .txt
    files with no header are also supported: the whole file is
    treated as body content, and the filename is used as the source
    label instead.
    """
    chunks = []
    for file_path in DOCS_DIR.glob("*.txt"):
        content = file_path.read_text(encoding="utf-8")

        source_url_match = re.search(r"SOURCE_URL:\s*(.+)", content)

        if source_url_match and "\n---\n" in content:
            # has the header format - split it off and use the real source URL
            _, _, body = content.partition("---")
            source_url = source_url_match.group(1).strip()
        else:
            # plain file, no header - use the whole file as body content
            body = content
            source_url = file_path.name

        paragraphs = [p.strip() for p in body.split("\n\n") if len(p.strip()) > 15]
        for paragraph in paragraphs:
            chunks.append({"text": paragraph, "source_url": source_url})

    return chunks


def ingest_documents():
    """
    Load, chunk, embed, and store all local documents into Qdrant.
    Safe to re-run - it recreates the collection each time.
    """
    chunks = _load_and_chunk_documents()
    if not chunks:
        print(f"No documents found in {DOCS_DIR}")
        return

    texts = [c["text"] for c in chunks]
    response = cohere_client.embed(
        texts=texts,
        model=COHERE_MODEL,
        input_type="search_document",
    )
    embeddings = response.embeddings

    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"text": chunk["text"], "source_url": chunk["source_url"]},
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]

    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Ingested {len(points)} chunks from {DOCS_DIR} into Qdrant.")


def search_context(query: str, top_k: int = 2) -> list[dict]:
    """
    Embed the query and return the top_k most relevant chunks,
    each with its text and source URL for attribution.
    """
    response = cohere_client.embed(
        texts=[query],
        model=COHERE_MODEL,
        input_type="search_query",
    )
    query_embedding = response.embeddings[0]

    hits = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
    )

    return [
        {"text": hit.payload["text"], "source_url": hit.payload["source_url"]}
        for hit in hits
    ]


if __name__ == "__main__":
    # python3 -m app.rag
    ingest_documents()

    print("\nTest search:")
    results = search_context("misleading claims about financial returns")
    for r in results:
        print(f"- {r['text']}\n  (source: {r['source_url']})\n")