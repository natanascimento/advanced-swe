import os
import pandas as pd
import numpy as np
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection


from semantic_search.core.config import settings

pd.set_option('display.max_columns', None)

def _load_synthetichealth(path):
    df = pd.concat([pd.read_json(file) for file in path])
    df = pd.json_normalize(df['entry'], max_level=10)
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df = df.explode(col)

    normalized_dfs = []

    for col in df.columns:
        # Check if the column contains dictionaries
        if df[col].apply(lambda x: isinstance(x, dict)).any():
            # Normalize the column and prefix the keys with the column name
            normalized_df = pd.json_normalize(df[col]).add_prefix(f'{col}.')
            # Add the normalized DataFrame to the list
            normalized_dfs.append(normalized_df)
            # Drop the original column from the DataFrame
            df = df.drop(columns=[col])
    df = pd.concat([df] + normalized_dfs)
    return df.apply(lambda row: ' '.join(row.astype(str)), axis=1).tolist()

def _similarity_search(collection: Collection, query_embedding: list):

    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()
    
    if not isinstance(query_embedding[0], list):
        query_embedding = [query_embedding]
    
    query_embedding = [[float(x) for x in embedding] for embedding in query_embedding]

    # Debugging: Print the query_embedding to ensure it is correctly formatted
    print(f"Query Embedding (first 10 elements): {query_embedding[0][:10]}")
    print(f"Query Embedding : {query_embedding}")
    print(f"Type of first element: {type(query_embedding[0][0])}")

    # Realizar a consulta de similaridade
    try:
        results = collection.search(data=query_embedding, 
                                    anns_field="embedding",
                                    param={"metric_type": "L2", "params": {"nprobe": 10}},
                                    limit=5)
        # Mostrar os resultados da consulta
        for result in results:
            print(f"Id: {result.id}, Distância: {result.distance}")
    except Exception as e:
        print(f"Erro durante a busca: {e}")

def start():
    path = f"{settings.app.DATALAKE_PATH}/raw/synthetichealth"
    files = [f"{path}/{file}" for file in os.listdir(path)]
    df = _load_synthetichealth(path=files)

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(df, show_progress_bar=True, device="cuda")

    embeddings = embeddings.astype(float).tolist()

    print(len(embeddings))

    connections.connect("default", host="localhost", port="19530")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
    ]
    schema = CollectionSchema(fields, description="Minha coleção de embeddings")

    collection = Collection(name="synthetichealth", schema=schema)
    collection.create_index(field_name="embedding")

    collection.insert([embeddings])
    collection.load()


    _similarity_search(collection=collection, query_embedding=embeddings)

