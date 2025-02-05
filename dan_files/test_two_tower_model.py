import pytest
import torch
import torch.nn as nn
from two_towers_model import TwoTowerModel, compute_tower_loss


@pytest.fixture
def model_params():
    vocab_size = 100
    emb_dims = 16
    embedding_matrix = torch.randn(vocab_size, emb_dims)
    return {
        'vocab_size': vocab_size,
        'emb_dims': emb_dims,
        'embedding_matrix': embedding_matrix,
        'query_tower_layers': 2,
        'answer_tower_layers': 2,
        'query_tower_hidden_dim': 32,
        'answer_tower_hidden_dim': 32
    }


@pytest.fixture
def batch_inputs(model_params, batch_size=4):
    """
    Creates a batch of variable-length input sequences for testing.
    
    Args:
        model_params: Dictionary containing model parameters
        batch_size: Number of sequences to generate
    
    Returns:
        tuple: (query_inputs, pos_answer_inputs, neg_answer_inputs)
        Each element is a list of 1-D tensors representing sequences
    """
    query_inputs = [
        torch.randint(0, model_params['vocab_size'], (torch.randint(3, 6, (1,)).item(),))
        for _ in range(batch_size)
    ]
    pos_answer_inputs = [
        torch.randint(0, model_params['vocab_size'], (torch.randint(3, 6, (1,)).item(),))
        for _ in range(batch_size)
    ]
    neg_answer_inputs = [
        torch.randint(0, model_params['vocab_size'], (torch.randint(3, 6, (1,)).item(),))
        for _ in range(batch_size)
    ]
    
    return query_inputs, pos_answer_inputs, neg_answer_inputs

@pytest.fixture
def model(model_params):
    return TwoTowerModel(**model_params)

def test_model_initialization(model_params):
    """Test if model initializes correctly with given configuration"""
    model = TwoTowerModel(**model_params)
    
    # Test 1: Check if embedding layer is frozen
    assert model.embedding.weight.requires_grad == False
    
    # Test 2: Verify query tower structure
    expected_query_layers = model_params['query_tower_layers'] * 2  # *2 because each layer has Linear + ReLU
    actual_query_layers = len([m for m in model.query_tower if isinstance(m, (nn.Linear, nn.ReLU))])
    assert actual_query_layers == expected_query_layers
    
    # Test 3: Verify answer tower structure
    expected_answer_layers = model_params['answer_tower_layers'] * 2
    actual_answer_layers = len([m for m in model.answer_tower if isinstance(m, (nn.Linear, nn.ReLU))])
    assert actual_answer_layers == expected_answer_layers
    
    # Test 4: Check final layer dimensions
    query_final_layer = [m for m in model.query_tower if isinstance(m, nn.Linear)][-1]
    assert query_final_layer.out_features == model_params['emb_dims']


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
def test_output_dimensions(model, model_params, batch_inputs, batch_size):
    """Test if output dimensions are correct for various batch sizes"""
    query_inputs, pos_inputs, neg_inputs = batch_inputs  # Use fixture directly
    
    query_output, pos_output, neg_output = model(query_inputs, pos_inputs, neg_inputs)
    
    assert query_output.shape == (len(query_inputs), model_params['emb_dims'])
    assert pos_output.shape == (len(pos_inputs), model_params['emb_dims'])
    assert neg_output.shape == (len(neg_inputs), model_params['emb_dims'])

def test_embedding_freeze(model, model_params, batch_inputs):
    """Test if embeddings remain frozen during backward pass"""
    # Store initial embedding weights
    initial_weights = model.embedding.weight.clone()
    
    # Get inputs from fixture
    query_inputs, pos_answer_inputs, neg_answer_inputs = batch_inputs
    
    query_output, pos_output, neg_output = model(query_inputs, pos_answer_inputs, neg_answer_inputs)
    
    # Compute loss
    loss = compute_tower_loss(query_output, pos_output, neg_output)
    loss.backward()
    
    # Check if embedding weights remained unchanged
    assert torch.allclose(model.embedding.weight, initial_weights)

def test_output_values_range(model, model_params, batch_inputs):
    """Test if output values are in a reasonable range after ReLU"""
    query_inputs, pos_answer_inputs, neg_answer_inputs = batch_inputs
    
    query_output, pos_output, neg_output = model(query_inputs, pos_answer_inputs, neg_answer_inputs)
    
    # Check if all values are non-negative (due to ReLU)
    assert (query_output >= 0).all()
    assert (pos_output >= 0).all()
    assert (neg_output >= 0).all()

@pytest.mark.parametrize("query_layers,answer_layers", [
    (1, 1),
    (2, 3),
    (3, 2),
    (4, 4)
])
def test_different_tower_depths(model_params, query_layers, answer_layers, batch_inputs):
    """Test if model works with different tower depths"""
    model_params['query_tower_layers'] = query_layers
    model_params['answer_tower_layers'] = answer_layers
    
    model = TwoTowerModel(**model_params)
    query_inputs, pos_answer_inputs, neg_answer_inputs = batch_inputs
    
    query_output, pos_output, neg_output = model(query_inputs, pos_answer_inputs, neg_answer_inputs)
    
    assert query_output.shape == (len(query_inputs), model_params['emb_dims'])
    assert pos_output.shape == (len(pos_answer_inputs), model_params['emb_dims'])
    assert neg_output.shape == (len(neg_answer_inputs), model_params['emb_dims'])

# New test for loss function
def test_loss_computation():
    """Test if loss behaves correctly with known similarities"""
    batch_size = 2
    emb_dim = 3
    margin = 0.5
    
    # Create embeddings that will have known cosine similarities
    query_emb = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float)
    pos_emb = torch.tensor([[0.866, 0.5, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float)  # ~30° and 90° from query
    neg_emb = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float)   # 90° and 180° from query
    
    loss = compute_tower_loss(query_emb, pos_emb, neg_emb, margin=margin)
    
    # First pair: pos_sim ≈ 0.866, neg_sim = 0
    # Second pair: pos_sim = 0, neg_sim = -1
    # Both should satisfy the margin, so loss should be 0
    assert loss.item() == 0.0

    # Now test with more similar negative samples
    neg_emb_close = torch.tensor([[0.966, 0.259, 0.0], [0.866, 0.5, 0.0]], dtype=torch.float)  # ~15° from query
    loss_with_close_neg = compute_tower_loss(query_emb, pos_emb, neg_emb_close, margin=margin)
    
    # Loss should be positive as negative samples are too similar to query
    assert loss_with_close_neg.item() > 0.0

