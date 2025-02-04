# Imports
from typing import Union, List
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import re
import more_itertools
import torch
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import random

def preprocess_text(text_input: Union[str, List[str]], min_freq: int = 5) -> tuple[List[str], dict]:
    """
    Preprocess text input and create vocabulary.
    """

    # Initialise Lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Define special tokens using a marker that word_tokenize won't split
    SPECIAL_TOKENS = {
        '.': 'XPERIODX',
        ',': 'XCOMMAX',
        '"': 'XQUOTATION_MARKX',
        ';': 'XSEMICOLONX',
        '!': 'XEXCLAMATION_MARKX',
        '?': 'XQUESTION_MARKX',
        '(': 'XLEFT_PARENX',
        ')': 'XRIGHT_PARENX',
        '--': 'XHYPHENSX',
        ':': 'XCOLONX',
        "'": 'XAPOSTROPHEX'
    }
    
    # Mapping for restoring angle brackets
    RESTORE_TOKENS = {
        f'XPERIODX': '<PERIOD>',
        f'XCOMMAX': '<COMMA>',
        f'XQUOTATION_MARKX': '<QUOTATION_MARK>',
        f'XSEMICOLONX': '<SEMICOLON>',
        f'XEXCLAMATION_MARKX': '<EXCLAMATION_MARK>',
        f'XQUESTION_MARKX': '<QUESTION_MARK>',
        f'XLEFT_PARENX': '<LEFT_PAREN>',
        f'XRIGHT_PARENX': '<RIGHT_PAREN>',
        f'XHYPHENSX': '<HYPHENS>',
        f'XCOLONX': '<COLON>',
        f'XAPOSTROPHEX': '<APOSTROPHE>'
    }
    
    def clean_text(text: str) -> str:
        # Convert to lowercase
        text = text.lower()
        
        # Replace special characters with temporary tokens
        for char, token in SPECIAL_TOKENS.items():
            text = text.replace(char, f' {token} ')
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^a-zA-Z0-9\s_X]', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    # Convert input to list of sentences
    if isinstance(text_input, str):
        sentences = sent_tokenize(text_input)
    elif isinstance(text_input, list):
        sentences = []
        for line in text_input:
            if isinstance(line, str):
                try:
                    sentences.extend(sent_tokenize(line))
                except:
                    sentences.append(line)
    else:
        raise ValueError("Input must be either a string or list of strings")

    # Process all sentences
    processed_words = []
    for sentence in sentences:
        # Clean the text
        cleaned_text = clean_text(sentence)
        # Tokenize
        try:
            words = word_tokenize(cleaned_text)
            # Restore angle bracket format
            words = [RESTORE_TOKENS.get(word, word) for word in words]
            # Apply lemmatization only to non-special tokens
            words = [word if word in RESTORE_TOKENS.values() 
                    else lemmatizer.lemmatize(word) 
                    for word in words]
        except:
            words = cleaned_text.split()
        processed_words.extend(words)

    # Count word frequencies
    word_counts = Counter(processed_words)
    
    # Create vocabulary (only including words that meet minimum frequency)
    vocab = [word for word, count in word_counts.items() if count >= min_freq]
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    # Filter words based on vocabulary
    processed_words = [word for word in processed_words if word in word_to_idx]
    
    return processed_words, word_to_idx

def create_skipgram_pairs(processed_words: List[str], 
                         word_to_idx: dict, 
                         context_len: int = 3) -> tuple[List[int], List[List[int]]]:
    """
    Create skipgram pairs from a list of processed words.
    """
    if context_len % 2 == 0:
        raise ValueError("context_len should be an odd number")
        
    window_radius = context_len // 2
    input_indices = []
    context_indices = []
    vocab_size = len(word_to_idx)  # Added this line
    
    # Use sliding window to create pairs
    windows = list(more_itertools.windowed(processed_words, context_len))
    
    for window in windows:
        if None in window:  # Skip incomplete windows
            continue
            
        # Get center word and context
        center_word = window[window_radius]
        context = list(window[:window_radius]) + list(window[window_radius + 1:])
        
        # Only add if all words are in vocabulary AND indices are within range
        if center_word in word_to_idx and all(w in word_to_idx for w in context):
            center_idx = word_to_idx[center_word]
            context_idxs = [word_to_idx[w] for w in context]
            
            # Added this check
            if center_idx < vocab_size and all(idx < vocab_size for idx in context_idxs):
                input_indices.append(center_idx)
                context_indices.append(context_idxs)
    
    return input_indices, context_indices

def evaluate_accuracy(model, X, Y, vocab_size):
    """
    Calculate accuracy by comparing positive sample scores with negative sample scores.
    We want positive context words to have higher scores than random negative words.
    """
    model.eval()
    
    with torch.no_grad():
        # Get embeddings for all input words
        emb = model.embeddings(X)  # [num_samples, embedding_dim]
        
        # Get positive context embeddings
        pos_ctx = model.output_weights.weight[Y]  # [num_samples, 2, embedding_dim]
        
        # Generate negative samples
        neg_samples = torch.randint(0, vocab_size, Y.shape)
        neg_ctx = model.output_weights.weight[neg_samples]
        
        # Calculate similarity scores
        emb_reshaped = emb.unsqueeze(-1)  # [num_samples, embedding_dim, 1]
        pos_scores = torch.bmm(pos_ctx, emb_reshaped).squeeze(-1)  # [num_samples, 2]
        neg_scores = torch.bmm(neg_ctx, emb_reshaped).squeeze(-1)  # [num_samples, 2]
        
        # Accuracy: how often positive scores > negative scores
        accuracy = (pos_scores > neg_scores).float().mean().item()
        
    return accuracy

def summarise_dataset(X, Y, vocab_size):
    # After creating the dataset
    print("Dataset size:", len(X))
    print("Vocabulary size:", vocab_size)
    print("Max index in X:", X.max().item())
    print("Max index in Y:", Y.max().item())
    print("Sample X values:", X[:10])
    print("Sample Y values:", Y[:10])

def plot_loss(lossi):
    """
    Plot loss values using PyTorch operations.
    """
    losses = torch.tensor(lossi)
    window_size = max(len(losses) // 100, 1)
    
    # Reshape and mean
    remainder = len(losses) % window_size
    if remainder:
        # Pad with the last value to make it evenly divisible
        padding = window_size - remainder
        losses = torch.cat([losses, losses[-1].repeat(padding)])
    
    averaged_losses = losses.view(-1, window_size).mean(1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(averaged_losses)
    plt.title('Training Loss Over Time')
    plt.xlabel(f'Steps (averaged over {window_size} steps)')
    plt.ylabel('Loss')
    plt.show()

def get_nearest_neighbors(word, word_to_idx, embeddings, n=5):
    """
    Find the top-n nearest neighbors of a word in the embedding space.
    
    Args:
        word (str): The target word.
        stoi (dict): Mapping from words to indices.
        embeddings (torch.Tensor): Learned word embeddings (shape: V x d).
        n (int): Number of nearest neighbors to retrieve.
    
    Returns:
        List of tuples (neighbor_word, similarity_score).
    """
    if word not in word_to_idx:
        return f"'{word}' not in vocabulary."
    
    word_idx = word_to_idx[word]
    word_embedding = embeddings[word_idx].unsqueeze(0)  # Shape: 1 x d
    
    # Compute cosine similarity between the target embedding and all embeddings
    similarities = cosine_similarity(word_embedding.detach().numpy(), embeddings.detach().numpy())
    similarities = similarities[0]  # Flatten
    
    # Get top-n similar words (excluding the word itself)
    nearest_indices = similarities.argsort()[-n-1:][::-1][1:]  # Exclude the word itself
    nearest_words = [(list(word_to_idx.keys())[idx], similarities[idx]) for idx in nearest_indices]
    
    print("The nearest neighbours of ", word, " are: ", nearest_words)
    return nearest_words

def get_random_sample(processed_words, n=20):
    # Using random
    # Make a copy so you don't shuffle the original list
    words_sample = processed_words.copy()
    random.shuffle(words_sample)

    # Print first 10 shuffled words
    print("Random sample of words:")
    print(words_sample[:20])
    return words_sample

def get_word_indices(words: List[str], word_to_idx: dict) -> List[int]:
    """
    Get indices for a list of words, with error handling for unknown words.
    
    Args:
        words: List of words to look up
        word_to_idx: Dictionary mapping words to indices
    
    Returns:
        List of indices for the input words. Unknown words will be noted.
    """
    indices = []
    unknown_words = []
    
    for word in words:
        if word in word_to_idx:
            indices.append(word_to_idx[word])
        else:
            unknown_words.append(word)
    
    # Uncomment below to print unknown words
    # if unknown_words:
    #     print(f"Warning: Words not in vocabulary: {unknown_words}")
    
    return indices