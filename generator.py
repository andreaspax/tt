import datasets
import random
import pandas as pd
import word2vec
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
    
    flat_df['unrelated_passage'] = flat_df.apply(get_unrelated_passage, axis=1)

    return flat_df


def process_dataframe(df, text_columns):
    vocab = tokeniser.load_vocab('vocab.json')
    """Add embeddings for all text columns"""
    for col in text_columns:
        # Create embedding column
        df[f'{col}_embedding'] = df[col].apply(word2vec.text_to_embeddings(df[col], vocab, 'id_vectors.npy'))
        
        # Convert numpy arrays to lists for storage
        df[f'{col}_embedding'] = df[f'{col}_embedding'].apply(lambda x: x.tolist())
    
    return df

    

ds_test = datasets.Dataset.from_parquet("ms_marco_test.parquet")
# # Create flattened DataFrame
flat_test = flatten_dataset(ds_test).drop(columns=['passages.url', 'answers', 'query_type','wellFormedAnswers','passages.is_selected','query_id'], errors='ignore')
print(flat_test.head(20))

proc_test = process_dataframe(flat_test, ['query', 'passages.passage_text', 'unrelated_passage'])
print(proc_test.head(20))
