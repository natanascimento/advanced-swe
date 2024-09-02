import os
import pandas as pd
import numpy as np

from datasets import Dataset
from sentence_transformers import SentenceTransformer
from pymilvus import (connections, CollectionSchema, 
                      FieldSchema, DataType, Collection)

from semantic_search.core.config import settings
from semantic_search.core.logging import logger

pd.set_option('display.max_columns', None)


def __load_synthetichealth(path):
    logger.info("Loading data")
    df = pd.concat([pd.read_json(file) for file in path])
    df = pd.json_normalize(df['entry'], max_level=10)

    logger.info("Exploding each colum if that column is a type of list")
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df = df.explode(col)
    
    normalized_dfs = []

    logger.info("Normalize data")
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, dict)).any():
            normalized_df = pd.json_normalize(df[col]).add_prefix(f'{col}.')
            normalized_dfs.append(normalized_df)
            df = df.drop(columns=[col])

    logger.info("Integrate data")
    df = pd.concat([df] + normalized_dfs)

    logger.info("Return data normalized (list exploded and structs flattened)")
    return df.apply(lambda row: ' '.join(row.astype(str)), axis=1).tolist()


def __similarity_search(collection: Collection, query_embedding: list):

    if isinstance(query_embedding, np.ndarray):
        logger.info("Transform np.ndarray into list")
        query_embedding = query_embedding.tolist()
    
    if not isinstance(query_embedding[0], list):
        logger.info("Adding query_embedding to list")
        query_embedding = [query_embedding]
    
    logger.info("Generating query embedding")
    query_embedding = [[float(x) for x in embedding] for embedding in query_embedding]

    try:
        logger.info("Return results")
        results = collection.search(data=query_embedding, 
                                    anns_field="embedding",
                                    param={"metric_type": "L2", "params": {"nprobe": 10}},
                                    limit=5)
        for hits in results:
            for hit in hits:
                logger.info(f"ID: {hit.id} | Distance: {hit.distance}")
    except Exception as e:
        logger.error(f"Erro durante a busca: {e}")

def _milvus_insert(embeddings, base: str):
    logger.info("Connecting to milvus")
    connections.connect(settings.milvus.db,
                        host=settings.milvus.host,
                        port=settings.milvus.port)
    logger.info("Connection stablished")

    fields = [
        FieldSchema(name="id",
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=True),
        FieldSchema(name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=embeddings.shape[1])
    ]
    logger.info("Creating collection")
    schema = CollectionSchema(fields,
                              description="Collection related to synthetichealth source")
    collection = Collection(name=base,
                            schema=schema)

    collection.create_index(field_name="embedding",
                            index_params={"metric_type": "L2",
                                          "index_type": "IVF_FLAT",
                                          "params": {"nlist": 1024}})
        
    logger.info("Loading fields")
    collection.load()

    logger.info("Inserting data")
    collection.insert([embeddings.tolist()])

    return collection


def start():
    base = "synthetichealth"

    path = f"{settings.app.DATALAKE_PATH}/raw/{base}"
    files = [f"{path}/{file}" for file in os.listdir(path)]

    sentences = __load_synthetichealth(path=files)

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(sentences, show_progress_bar=True, device="cuda")

    collection = _milvus_insert(embeddings, base=base)

    logger.info("Querying milvus vector database")
    query_sentence = "Phone Number that start with 555"
    query_embedding = model.encode([query_sentence])

    __similarity_search(collection=collection, query_embedding=query_embedding)