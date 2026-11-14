"""Verify Ollama embeddings and read-only Milvus search for DAY07."""
import json
import os
from contextlib import closing
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from pymilvus import MilvusClient


def main():
    load_dotenv()
    ollama = os.environ["OLLAMA_URL"].rstrip("/")
    model = os.environ["OLLAMA_EMBED_MODEL"]
    collection = os.environ["MILVUS_COLLECTION"]
    request = Request(
        f"{ollama}/api/embed",
        json.dumps({"model": model, "input": "GR00T N1.7 fine-tuning GPU memory requirements"}).encode(),
        {"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=60) as response:
        vector, = json.load(response)["embeddings"]
    if len(vector) != 1024:
        raise RuntimeError(f"Expected 1024 dimensions for the DAY06 index, got {len(vector)}.")
    print(f"Ollama: {model}, {len(vector)} dimensions")

    with closing(MilvusClient(os.environ["MILVUS_URI"], timeout=15)) as client:
        client.load_collection(collection)
        count = client.query(collection, output_fields=["count(*)"])[0]["count(*)"]
        hits = client.search(collection, [vector], limit=3,
                             output_fields=["document_id", "page"])[0]
        if len(hits) != 3:
            raise RuntimeError(f"Expected three hits, got {len(hits)}.")
        print(f"Milvus {client.get_server_version()}: {collection}, {count} chunks")
        for hit in hits:
            print(f'{hit["distance"]:.4f} {hit["entity"]["document_id"]}')


if __name__ == "__main__":
    main()
