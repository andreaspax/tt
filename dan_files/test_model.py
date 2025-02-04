import torch
import pytest
import torch.nn as nn
from model import Word2Vec

# Import your model class; adjust the import as needed.
# from your_module import Word2Vec


# 1. Test that the forward pass produces a scalar loss with valid input.
def test_forward_output_shape():
    vocab_size = 10
    emb_dims = 5
    model = Word2Vec(vocab_size, emb_dims)
    
    # Create dummy data with indices within the range [0, vocab_size-1].
    # For example, let x be a mini-batch of 3 token indices.
    x = torch.tensor([1, 2, 3])
    positive_samples = torch.tensor([2, 3, 4])
    negative_samples = torch.tensor([5, 6, 7])
    
    loss = model(x, positive_samples, negative_samples)
    
    # The loss should be a 0-dimensional tensor (a scalar).
    assert loss.dim() == 0, "Expected scalar loss output, got a tensor with shape: " + str(loss.shape)
    # Optionally, check that the loss is a finite number.
    assert torch.isfinite(loss), "Loss is not finite."

# 2. Test that out-of-range indices cause a ValueError.
def test_out_of_range_indices():
    vocab_size = 10
    emb_dims = 5
    model = Word2Vec(vocab_size, emb_dims)
    
    # Create an input tensor with an out-of-range index.
    x = torch.tensor([9, 10])  # Here, 10 is out of range (valid indices: 0 to 9)
    positive_samples = torch.tensor([1, 2])
    negative_samples = torch.tensor([3, 4])
    
    with pytest.raises(ValueError) as excinfo:
        _ = model(x, positive_samples, negative_samples)
    assert "Input indices must be less than vocab_size" in str(excinfo.value)

# 3. Test with known inputs to validate computations.
def test_forward_computation_consistency():
    vocab_size = 5
    emb_dims = 3
    model = Word2Vec(vocab_size, emb_dims)
    
    # Initialize embeddings and weights to fixed values for reproducibility.
    torch.manual_seed(0)
    model.embeddings.weight.data = torch.arange(vocab_size * emb_dims, dtype=torch.float32).view(vocab_size, emb_dims)
    model.output_weights.weight.data = torch.arange(vocab_size * emb_dims, dtype=torch.float32).view(vocab_size, emb_dims)
    
    # Define inputs with valid indices.
    x = torch.tensor([0, 1])
    positive_samples = torch.tensor([2, 3])
    negative_samples = torch.tensor([1, 0])
    
    loss = model(x, positive_samples, negative_samples)
    # Check that the loss is computed and is a float value.
    assert isinstance(loss.item(), float)
    # You can also set an expected loss range (or even a specific value) if you have computed it manually.
    assert loss.item() > 0, "Loss should be positive."

# Run the tests with: pytest test_word2vec.py
