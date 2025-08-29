import os
import time
from openai import OpenAI
from typing import List
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# --- Initialize the OpenAI Client ---
# It will automatically look for the OPENAI_API_KEY environment variable
try:
    client = OpenAI()
    # Let's use the new, cheaper, and highly effective model
    # You can swap this for "text-embedding-3-large" for potentially higher quality at a higher cost
    EMBEDDING_MODEL_NAME = "text-embedding-3-small"
    print(f"OpenAI client initialized. Using model: {EMBEDDING_MODEL_NAME}")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Please ensure your OPENAI_API_KEY is set in your .env file.")
    client = None

def generate_embeddings(texts: List[str]) -> List[np.ndarray]:
    """
    Generates embeddings for a list of texts using the OpenAI API.
    Handles batching and retries for API robustness.
    """
    if not client or not texts:
        return []

    all_embeddings = []
    # OpenAI API has a limit on batch size, 2048 is a safe number.
    batch_size = 1000 
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"Generating embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...")
        
        try:
            # Replace newlines, as they can sometimes cause issues with the API
            batch = [text.replace("\n", " ") for text in batch]
            
            response = client.embeddings.create(
                input=batch,
                model=EMBEDDING_MODEL_NAME
            )
            
            # Extract embeddings from the response
            batch_embeddings = [np.array(embedding_data.embedding) for embedding_data in response.data]
            all_embeddings.extend(batch_embeddings)
            
            # OpenAI has rate limits, a small sleep can help avoid them on large jobs
            time.sleep(1) 

        except Exception as e:
            print(f"An error occurred with OpenAI API: {e}")
            # For a production system, you might add more sophisticated retry logic here.
            # For now, we'll append empty arrays and continue.
            print(f"Failed to process batch starting at index {i}. Appending empty embeddings for this batch.")
            all_embeddings.extend([np.array([]) for _ in batch])
            continue
            
    print(f"Embedding generation complete. Generated {len(all_embeddings)} vectors.")
    return all_embeddings