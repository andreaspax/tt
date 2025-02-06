import datasets
import random
import pandas as pd
import numpy as np
import tokeniser

# Convert to pandas and flatten nested structures
def flatten_dataset(dataset):
    df = dataset.to_pandas()

    # Flatten the 'passages' column containing dicts with lists
    passages_df = pd.json_normalize(df['passages'].tolist())
    passages_df.columns = [f'passages.{col}' for col in passages_df.columns]
    
    # Combine with original data
    flat_df = pd.concat([df.drop('passages', axis=1), passages_df], axis=1)
    
    # Explode 'passages.is_selected' and 'passages.passage_text' to show each item with repeating query and query_id values
    flat_df = flat_df.explode('passages.passage_text').reset_index(drop=True)
    
    # Create pool of all passages with their query_ids
    all_passages = list(zip(flat_df['query_id'], flat_df['passages.passage_text']))
    
    # Add random unrelated passage from different query
    def get_unrelated_passage(row):
        while True:
            # Randomly select until we find passage from different query
            random_query_id, random_passage = random.choice(all_passages)
            if random_query_id != row['query_id']:
                return random_passage
    
    flat_df['document_unrelated'] = flat_df.apply(get_unrelated_passage, axis=1)
    flat_df = flat_df.drop(columns=['passages.url', 'answers', 'query_type',
                                  'wellFormedAnswers','passages.is_selected',
                                  'query_id'], errors='ignore')    
    flat_df['document_related'] = flat_df['passages.passage_text']

    return flat_df


class TripleGenerator:
    def __init__(self):
        self.triples = []

    # Modified method signature with proper self reference
    def generate_triples(self, dataset):
        self.triples = []
        for _, row in dataset.iterrows():
            query = row['query']
            related = row['document_related']
            unrelated = row['document_unrelated']
            self.triples.append((query, related, unrelated))
        return self.triples

vocab = tokeniser.load_vocab("vocab.json")

def tokenize_triples(triples, vocab):
    tokenized_triples = []
    for triple in triples:
        tokenized_query = tokeniser.text_to_ids(triple[0], vocab)
        tokenized_relevant_doc = tokeniser.text_to_ids(triple[1], vocab)
        tokenized_irrelevant_doc = tokeniser.text_to_ids(triple[2], vocab)
        tokenized_triples.append((tokenized_query, tokenized_relevant_doc, tokenized_irrelevant_doc))
    return tokenized_triples


# Load pre-trained ID-to-embedding mappings
id_vectors = np.load('id_vectors.npy', allow_pickle=True).item() # The allow_pickle parameter determines whether the function is allowed to use Python's pickle module to serialize or deserialize data.

def ids_to_embeddings(token_ids: list[int], id_vectors: dict) -> list[np.array]:
    return [id_vectors.get(token_id, np.zeros(300)) for token_id in token_ids]

def embed_triples(tokenized_triples, id_vectors):
    embedded_triples = []
    for tokenized_query, tokenized_relevant_doc, tokenized_irrelevant_doc in tokenized_triples:
        embedded_query = ids_to_embeddings(tokenized_query, id_vectors)
        embedded_relevant_doc = ids_to_embeddings(tokenized_relevant_doc, id_vectors)
        embedded_irrelevant_doc = ids_to_embeddings(tokenized_irrelevant_doc, id_vectors)
        embedded_triples.append((embedded_query, embedded_relevant_doc, embedded_irrelevant_doc))
    return embedded_triples

def average_embeddings(embedded_triples):
    averaged_triples = []
    
    for embedded_query, embedded_relevant_doc, embedded_irrelevant_doc in embedded_triples:
        
        # Calculate the mean embedding for each component
        # np.mean to calculate the means of the embeddings
        avg_query = np.mean(embedded_query, axis=0)
        avg_relevant_doc = np.mean(embedded_relevant_doc, axis=0)
        avg_irrelevant_doc = np.mean(embedded_irrelevant_doc, axis=0)
        
        # Append the averaged embeddings to the list
        averaged_triples.append((avg_query, avg_relevant_doc, avg_irrelevant_doc))
    
    return averaged_triples

if __name__ == "__main__":
    generator = TripleGenerator()

    # Process the test dataset
    print(f"Loading test parquet...")
    ds = datasets.Dataset.from_parquet("ms_marco_test.parquet")
    print(f"Flattening test parquet...")
    flat = flatten_dataset(ds)
    print(f"Converting to Triples test parquet...")
    triples = generator.generate_triples(flat)
    print(f"Tokenizing triples...")
    tokenized_triples = tokenize_triples(triples, vocab)
    print(f"Embedding triples...")
    embedded_triples = embed_triples(tokenized_triples, id_vectors)
    print(f"Averaging triples...")
    averaged_triples = average_embeddings(embedded_triples)
    print(f"Saving...")
    np.save('test_averaged_triples.npy', averaged_triples)

    # Process the train dataset
    print(f"Loading train parquet...")
    ds = datasets.Dataset.from_parquet("ms_marco_train.parquet")
    print(f"Flattening parquet...")
    flat = flatten_dataset(ds)
    print(f"Converting to Triples parquet...")
    triples = generator.generate_triples(flat)
    print(f"Tokenizing triples...")
    tokenized_triples = tokenize_triples(triples, vocab)
    print(f"Embedding triples...")
    embedded_triples = embed_triples(tokenized_triples, id_vectors)
    print(f"Averaging triples...")
    averaged_triples = average_embeddings(embedded_triples)
    print(f"Saving...")
    np.save('train_averaged_triples.npy', averaged_triples)

    # Process the train dataset
    print(f"Loading val parquet...")
    ds = datasets.Dataset.from_parquet("ms_marco_validation.parquet")
    print(f"Flattening parquet...")
    flat = flatten_dataset(ds)
    print(f"Converting to Triples parquet...")
    triples = generator.generate_triples(flat)
    print(f"Tokenizing triples...")
    tokenized_triples = tokenize_triples(triples, vocab)
    print(f"Embedding triples...")
    embedded_triples = embed_triples(tokenized_triples, id_vectors)
    print(f"Averaging triples...")
    averaged_triples = average_embeddings(embedded_triples)
    print(f"Saving...")
    np.save('validation_averaged_triples.npy', averaged_triples)

    

    