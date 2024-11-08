import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast
from openai import OpenAI
from config import load_config

# Load configuration values
config = load_config()
API_KEY = config['api_key']
MODEL = config['emb_model']
client = OpenAI(api_key=API_KEY)

# Load CSV and prepare the DataFrame once
df_test = pd.read_csv('testDataAllEmbedded16102024.csv', on_bad_lines='skip',
                 dtype={
                     'TestCode': str,
                     'TestName': str,
                     'TestNameEmbedding': str
                 })

df_ref = pd.read_csv('refDataAllEmbedded16102024.csv', on_bad_lines='skip',
                 dtype={
                     'RefCode': str,
                     'RefName': str,
                     'RefType': str,
                     'RefNameEmbedding': str
                 })

# Convert Embedding column from string representation to actual list of floats
df_test['TestNameEmbedding_new'] = df_test['TestNameEmbedding'].apply(lambda x: ast.literal_eval(x))
df_ref['RefNameEmbedding_new'] = df_ref['RefNameEmbedding'].apply(lambda x: ast.literal_eval(x))

# Function to generate embeddings using OpenAI
def get_embedding(text, model="text-embedding-3-large"):
    return client.embeddings.create(input=[text], model=model).data[0].embedding

# Function to find similar test based on cosine similarity
def map_test_code(input_test_name: str, threshold):
    # Generate embedding for the input test name
    input_embedding = get_embedding(input_test_name)
    # Convert embeddings from the DataFrame into a list of vectors
    # df_embeddings = np.array(df['TestNameEmbedding'].tolist())

    df_embeddings = df_test['TestNameEmbedding_new'].tolist()
    df_embeddings = np.array(df_embeddings)
    # Calculate cosine similarities between input embedding and all embeddings in the DataFrame
    similarities = cosine_similarity([input_embedding], df_embeddings)[0]

    # Get the index of the most similar test based on cosine similarity
    most_similar_index = np.argmax(similarities)
    highest_similarity = similarities[most_similar_index]

    # Check if the highest similarity exceeds the threshold
    if highest_similarity >= threshold:
        closest_test_name = df_test.iloc[most_similar_index]['TestName']
        closest_test_code = df_test.iloc[most_similar_index]['TestCode']
    else:
        closest_test_name = None
        closest_test_code = None

    return closest_test_name, closest_test_code

# Function to find similar ref name based on cosine similarity
def map_ref_code(input_ref_name: str, threshold):
    # Generate embedding for the input test name
    input_embedding = get_embedding(input_ref_name)

    # Convert embeddings from the DataFrame into a list of vectors
    df_embeddings = df_ref['RefNameEmbedding_new'].tolist()
    df_embeddings = np.array(df_embeddings)

    # Calculate cosine similarities between input embedding and all embeddings in the DataFrame
    similarities = cosine_similarity([input_embedding], df_embeddings)[0]

    # Get the index of the most similar ref based on cosine similarity
    most_similar_index = np.argmax(similarities)
    highest_similarity = similarities[most_similar_index]

    # Extract the RefName, RefCode and RefType from the DataFrame for the closest match
    # Check if the highest similarity exceeds the threshold
    if highest_similarity >= threshold:
        closest_ref_name = df_ref.iloc[most_similar_index]['RefName']
        closest_ref_code = df_ref.iloc[most_similar_index]['RefCode']
        closest_ref_type = df_ref.iloc[most_similar_index]['RefType']
    else:
        closest_ref_name = None
        closest_ref_code = None
        closest_ref_type = None

    return closest_ref_name, closest_ref_code, closest_ref_type
