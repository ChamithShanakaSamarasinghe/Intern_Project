from functools import lru_cache
from typing import List
from pymilvus import connections, Collection
from embedder import embed_query

# Milvus Connection
connections.connect(host="localhost", port="19530")

TEXT_COLLECTION_NAME = "text_chunks_collection"
IMAGE_COLLECTION_NAME = "image_embeddings_collection"

TEXT_COLLECTION = Collection(TEXT_COLLECTION_NAME)
IMAGE_COLLECTION = Collection(IMAGE_COLLECTION_NAME)

# Ensure collections are loaded (important for prod)
TEXT_COLLECTION.load()
IMAGE_COLLECTION.load()


# Caching Layer
@lru_cache(maxsize=256)
def cached_embed_query(query: str):
    """
    Cache embeddings for repeated queries
    """
    return embed_query(query)


# Retrieval
def retrieve_context(
    query: str,
    top_k: int = 3,
    max_distance: float = 0.8
) -> List[str]:
    """
    Retrieve relevant text chunks and images using vector similarity.
    Lower distance = higher similarity.
    """

    query_vector = cached_embed_query(query)

    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }

    results = []

    # -------- TEXT SEARCH --------
    text_hits = TEXT_COLLECTION.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["text"]
    )

    for hit in text_hits[0]:
        if hit.distance <= max_distance:
            results.append(hit.entity.get("text"))

    # -------- IMAGE SEARCH --------
    image_hits = IMAGE_COLLECTION.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["image_path"]
    )

    for hit in image_hits[0]:
        if hit.distance <= max_distance:
            results.append(f"Related image: {hit.entity.get('image_path')}")

    return results
