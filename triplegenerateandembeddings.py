#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:26:50 2025

@author: toluojo
"""

'''

FLOW OF OPERATIONS : 
    
- Generate triples (generate_triples()).
- Tokenize triples (tokenize_triples()).
- Embed tokenized triples (embed_triples()).
- Pad embedded triples (using pad_embeddings())

'''

import random
import pandas as pd
import collections
import re
import json
import numpy as np

# Load datasets from JSON
train_dataset = pd.read_json("train_dataset.json", orient="records", lines=True)
validation_dataset = pd.read_json("validation_dataset.json", orient="records", lines=True)
test_dataset = pd.read_json("test_dataset.json", orient="records", lines=True)

class TripleGenerator:
    def __init__(self):
        self.all_passages = []
        self.triples = []

    def extract_query(self, row):
        return row['query']

    def extract_passages(self, row):
        if isinstance(row['passages'], dict):
            return row['passages']['passage_text']
        return []

    def find_relevant_documents(self, row):
        if isinstance(row['passages'], dict):
            is_selected = row['passages']['is_selected']
            passage_texts = row['passages']['passage_text']
            return [passage_texts[i] for i, selected in enumerate(is_selected) if selected == 1]
        return []

    def sample_irrelevant_document(self, relevant_docs):
        if not self.all_passages:
            return None
        irrelevant_doc = random.choice(self.all_passages)
        while irrelevant_doc in relevant_docs:
            irrelevant_doc = random.choice(self.all_passages)
        return irrelevant_doc

    def generate_triples(self, dataset, dataset_name):
        self.triples = []
        for _, row in dataset.iterrows():
            query = self.extract_query(row)
            passages = self.extract_passages(row)
            self.all_passages.extend(passages)
            relevant_docs = self.find_relevant_documents(row)
            if not relevant_docs:
                continue
            for relevant_doc in relevant_docs:
                irrelevant_doc = self.sample_irrelevant_document(relevant_docs)
                if irrelevant_doc is None:
                    continue
                self.triples.append((query, relevant_doc, irrelevant_doc))
        return self.triples

# INSTANSIATING FUNCTIONS 
# Generate triples
triple_generator = TripleGenerator()
train_triples = triple_generator.generate_triples(train_dataset, "Train Dataset")
validation_triples = triple_generator.generate_triples(validation_dataset, "Validation Dataset")
test_triples = triple_generator.generate_triples(test_dataset, "Test Dataset")

#to save the triples to numpy arrays 
np.save('train_text_triples.npy', train_triples)
np.save('validation_text_triples.npy', validation_triples)
np.save('test_text_triples.npy', test_triples)



# Tokenizer functions
def load_vocab(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def text_to_ids(text: str, vocab: dict) -> list[int]:
    return [vocab.get(word.lower(), vocab['<UNK>']) for word in re.findall(r'\b\w+\b', text)]

# Load vocabulary
vocab = load_vocab("vocab.json")

def tokenize_triples(triples, vocab):
    tokenized_triples = []
    for triple in triples:
        tokenized_query = text_to_ids(triple[0], vocab)
        tokenized_relevant_doc = text_to_ids(triple[1], vocab)
        tokenized_irrelevant_doc = text_to_ids(triple[2], vocab)
        tokenized_triples.append((tokenized_query, tokenized_relevant_doc, tokenized_irrelevant_doc))
    return tokenized_triples



# Load pre-trained ID-to-embedding mappings
id_vectors = np.load('id_vectors.npy', allow_pickle=True).item() # The allow_pickle parameter determines whether the function is allowed to use Python's pickle module to serialize or deserialize data.

def ids_to_embeddings(token_ids: list[int], id_vectors: dict) -> list[np.array]:
    return [id_vectors.get(token_id, np.zeros(300)) for token_id in token_ids]

'''
Padding with zero vectors to ensure each sequence has the same length, so that they can be processed in parallel - for batch processing in nn
'''

# ENSURE HOMOGENOUS LENGTHS FOR EACH TRIPLE BY ADDING A PADDING 
# ENSURE THAT LIST OF TOKENISED WORDS/EMBEDDINGS ARE SAME LENGTH FOR EACH COMPONENTS 
# to process lengths of sequences in batches as nn requires all sequences in batch to be of the same length 
# here padding with ZERO vectors 
def pad_embeddings(embeddings, max_length, pad_value=np.zeros(300)):  # 300 is the embedding size
    return embeddings + [pad_value] * (max_length - len(embeddings)) if len(embeddings) < max_length else embeddings[:max_length]


# Convert tokenized triples to embeddings, this retruns a list of tuples - each embedding -  a list of NumPy arrays.
def embed_triples(tokenized_triples, id_vectors):
    embedded_triples = []
    for tokenized_query, tokenized_relevant_doc, tokenized_irrelevant_doc in tokenized_triples:
        embedded_query = ids_to_embeddings(tokenized_query, id_vectors)
        embedded_relevant_doc = ids_to_embeddings(tokenized_relevant_doc, id_vectors)
        embedded_irrelevant_doc = ids_to_embeddings(tokenized_irrelevant_doc, id_vectors)
        embedded_triples.append((embedded_query, embedded_relevant_doc, embedded_irrelevant_doc))
    return embedded_triples


# TOKENISE TRIPLES
train_tokenized_triples = tokenize_triples(train_triples, vocab)
validation_tokenized_triples = tokenize_triples(validation_triples, vocab)
test_tokenized_triples = tokenize_triples(test_triples, vocab)

# EMBED THESE TRIPLES 
train_embedded_triples = embed_triples(train_tokenized_triples, id_vectors)
validation_embedded_triples = embed_triples(validation_tokenized_triples, id_vectors)
test_embedded_triples = embed_triples(test_tokenized_triples, id_vectors)
# print(train_embedded_triples[:1]) # optional just to check -DEBUGGING 



#train_embedded_triples = np.array(train_embedded_triples)
#validation_embedded_triples = np.array(validation_embedded_triples)
#test_embedded_triples = np.array(test_embedded_triples)



# send triples as text form - test - to the git repo 

# do average poolimg 





'''

# Example usage: Pad each query, relevant doc, and irrelevant doc
max_length = 100  # Define max length of sequences, if sequence is shorter than this it will be padded , if longer it is truncated 
# the padding function is applied to each part of the triple - loop processes each triple in the list train_embedded_triples.
padded_train_triples = [(pad_embeddings(query, max_length), pad_embeddings(relevant_doc, max_length), pad_embeddings(irrelevant_doc, max_length))
                        for query, relevant_doc, irrelevant_doc in train_embedded_triples]


# PADDING VALIDATION TRIPLES 
padded_validation_triples = [(pad_embeddings(query, max_length), pad_embeddings(relevant_doc, max_length), pad_embeddings(irrelevant_doc, max_length))
    for query, relevant_doc, irrelevant_doc in validation_embedded_triples
]

# PADDING TEST TRIPLES 
padded_test_triples = [(pad_embeddings(query, max_length), pad_embeddings(relevant_doc, max_length), pad_embeddings(irrelevant_doc, max_length))
    for query, relevant_doc, irrelevant_doc in test_embedded_triples
]
'''


'''
print(np.array(padded_train_triples[0][0]).shape)  # Check shape of the first query in the first triple embedding size - (100,300)




# issue with consistent sizes of arrays - NEED TO ENSURE HOMOGENOUS LENGTHS FOR EACH TRIPLE - Ensure the list of tokenised words are the same length 
# Save padded triples as numpy arrays
#np.save('train_embedded_triples.npy', padded_train_triples)
#np.save('validation_embedded_triples.npy', padded_validation_triples)
np.save('test_embedded_triples.npy', padded_test_triples)

'''


























''' 
Triples : WORDS TO EMBEDDINGS 

import random
import pandas as pd
import collections
import re
import json
import numpy as np
import gensim.downloader as api

# Load datasets from JSON
train_dataset = pd.read_json("train_dataset.json", orient="records", lines=True)
validation_dataset = pd.read_json("validation_dataset.json", orient="records", lines=True)
test_dataset = pd.read_json("test_dataset.json", orient="records", lines=True)

class TripleGenerator:
    def __init__(self):
        self.all_passages = []
        self.triples = []

    def extract_query(self, row):
        return row['query']

    def extract_passages(self, row):
        if isinstance(row['passages'], dict):
            return row['passages']['passage_text']
        return []

    def find_relevant_documents(self, row):
        if isinstance(row['passages'], dict):
            is_selected = row['passages']['is_selected']
            passage_texts = row['passages']['passage_text']
            return [passage_texts[i] for i, selected in enumerate(is_selected) if selected == 1]
        return []

    def sample_irrelevant_document(self, relevant_docs):
        if not self.all_passages:
            return None
        irrelevant_doc = random.choice(self.all_passages)
        while irrelevant_doc in relevant_docs:
            irrelevant_doc = random.choice(self.all_passages)
        return irrelevant_doc

    def generate_triples(self, dataset, dataset_name):
        self.triples = []
        for _, row in dataset.iterrows():
            query = self.extract_query(row)
            passages = self.extract_passages(row)
            self.all_passages.extend(passages)
            relevant_docs = self.find_relevant_documents(row)
            if not relevant_docs:
                continue
            for relevant_doc in relevant_docs:
                irrelevant_doc = self.sample_irrelevant_document(relevant_docs)
                if irrelevant_doc is None:
                    continue
                self.triples.append((query, relevant_doc, irrelevant_doc))
        return self.triples

# Generate triples
triple_generator = TripleGenerator()
train_triples = triple_generator.generate_triples(train_dataset, "Train Dataset")
validation_triples = triple_generator.generate_triples(validation_dataset, "Validation Dataset")
test_triples = triple_generator.generate_triples(test_dataset, "Test Dataset")

# Tokenizer functions
def create_vocab(corpus: str, min_freq: int = 5) -> dict:
    tokens = re.findall(r'\b\w+\b', corpus.lower())
    word_counts = collections.Counter(tokens)
    filtered = {word: count for word, count in word_counts.items() if count >= min_freq}
    vocab = {word: idx for idx, word in enumerate(sorted(filtered), start=2)}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

def load_vocab(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def text_to_ids(text: str, vocab: dict) -> list[int]:
    return [vocab.get(word.lower(), vocab['<UNK>']) for word in re.findall(r'\b\w+\b', text)]

# Load vocabulary
vocab = load_vocab("vocab.json")

def tokenize_triples(triples, vocab):
    tokenized_triples = []
    for triple in triples:
        tokenized_query = text_to_ids(triple[0], vocab)
        tokenized_relevant_doc = text_to_ids(triple[1], vocab)
        tokenized_irrelevant_doc = text_to_ids(triple[2], vocab)
        tokenized_triples.append((tokenized_query, tokenized_relevant_doc, tokenized_irrelevant_doc))
    return tokenized_triples

train_tokenized_triples = tokenize_triples(train_triples, vocab)
validation_tokenized_triples = tokenize_triples(validation_triples, vocab)
test_tokenized_triples = tokenize_triples(test_triples, vocab)

# Load pre-trained embeddings
model = api.load('glove-wiki-gigaword-300')

# Create word embeddings for vocabulary
subset_vectors = {word: model[word] for word in vocab if word in model}
np.save('subset_vectors.npy', subset_vectors)

def text_to_embeddings(text: str, vocab: dict, vector_file: str) -> list[np.array]:
    vectors = np.load(vector_file, allow_pickle=True).item()
    token_ids = text_to_ids(text, vocab)
    return [vectors.get(token_id, np.zeros(300)) for token_id in token_ids]

# Convert tokenized triples to embeddings
def embed_triples(tokenized_triples, vocab, vector_file):
    embedded_triples = []
    for tokenized_query, tokenized_relevant_doc, tokenized_irrelevant_doc in tokenized_triples:
        embedded_query = text_to_embeddings(' '.join(map(str, tokenized_query)), vocab, vector_file)
        embedded_relevant_doc = text_to_embeddings(' '.join(map(str, tokenized_relevant_doc)), vocab, vector_file)
        embedded_irrelevant_doc = text_to_embeddings(' '.join(map(str, tokenized_irrelevant_doc)), vocab, vector_file)
        embedded_triples.append((embedded_query, embedded_relevant_doc, embedded_irrelevant_doc))
    return embedded_triples

train_embedded_triples = embed_triples(train_tokenized_triples, vocab, 'subset_vectors.npy')
validation_embedded_triples = embed_triples(validation_tokenized_triples, vocab, 'subset_vectors.npy')
test_embedded_triples = embed_triples(test_tokenized_triples, vocab, 'subset_vectors.npy')



so you generated code early combining the code for generating triples and tokenising them then using pretrained word2vec to map the ids to the embeddings but instead you mapped the words to the mebeddings i want you to redo the combination and just import the id_vectors.npy file . 

'''
