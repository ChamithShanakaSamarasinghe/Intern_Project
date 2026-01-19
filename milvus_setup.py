"""
milvus_setup.py

Purpose:
- Connect to Milvus
- Create collection for text embeddings
- Create ANN index
- Load collection into memory

This file is executed once during system setup.
"""

from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)

#Connect to Milvus
connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

print("Connected to Milvus successfully.")


#Define Constants
COLLECTION_NAME = "text_embeddings"
EMBEDDING_DIM = 768  


#Check if Collection Exists
if utility.has_collection(COLLECTION_NAME):
    print(f"Collection '{COLLECTION_NAME}' already exists.")
    collection = Collection(COLLECTION_NAME)
else:
    print(f"Creating collection '{COLLECTION_NAME}'...")

    #Define Collection Schema
    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=EMBEDDING_DIM
        ),
        FieldSchema(
            name="content",
            dtype=DataType.VARCHAR,
            max_length=1000
        ),
        FieldSchema(
            name="source",
            dtype=DataType.VARCHAR,
            max_length=255
        )
    ]

    schema = CollectionSchema(
        fields=fields,
        description="Text embeddings for RAG system"
    )

    collection = Collection(
        name=COLLECTION_NAME,
        schema=schema
    )

    print("Collection created successfully.")


#Create Index (ANN)
if not collection.has_index():
    print("Creating index on embedding field...")

    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }

    collection.create_index(
        field_name="embedding",
        index_params=index_params
    )

    print("Index created successfully.")
else:
    print("Index already exists.")

#Loading the Collection
collection.load()
print("Collection loaded into memory and ready for use.")
