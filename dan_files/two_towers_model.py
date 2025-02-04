import torch
import torch.nn as nn

class TwoTowerModel(nn.Module):
    def __init__(self, vocab_size, emb_dims, embedding_matrix, query_tower_layers, answer_tower_layers, 
                 query_tower_hidden_dim, answer_tower_hidden_dim, pad_token=0):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.pad_token = pad_token
        # Query tower - final layer outputs embedding dimension size for similarity comparison
        query_layers = []
        input_dim = emb_dims
        for i in range(query_tower_layers):
            if i == query_tower_layers - 1:
                query_layers.extend([
                    nn.Linear(input_dim, emb_dims),  # Final layer outputs same dim as embeddings
                    nn.ReLU()
                ])
            else:
                query_layers.extend([
                    nn.Linear(input_dim, query_tower_hidden_dim),
                    nn.ReLU()
                ])
                input_dim = query_tower_hidden_dim
        self.query_tower = nn.Sequential(*query_layers)
        
        # Answer tower - similar structure
        answer_layers = []
        input_dim = emb_dims
        for i in range(answer_tower_layers):
            if i == answer_tower_layers - 1:
                answer_layers.extend([
                    nn.Linear(input_dim, emb_dims),  # Final layer outputs same dim as embeddings
                    nn.ReLU()
                ])
            else:
                answer_layers.extend([
                    nn.Linear(input_dim, answer_tower_hidden_dim),
                    nn.ReLU()
                ])
                input_dim = answer_tower_hidden_dim
        self.answer_tower = nn.Sequential(*answer_layers)

    def pad_and_pool(self, sequences):
        """
        Takes a list of tensors (variable-length sequences), pads them, and returns
        the average pooled embeddings for each sequence.
        """
        # Check that none of the sequences is empty:
        for s in sequences:
            if s.numel() == 0:
                raise ValueError("Empty sequence encountered, please ensure all sequences contain at least one token.")
        
        # Pad sequences to create a tensor of shape (batch_size, max_seq_length)
        padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=self.pad_token)
        # Create mask: 1 for valid token, 0 for padding
        mask = (padded != self.pad_token).float()
        # Pass the padded data through embedding layer
        embedded = self.embedding(padded)  # (batch_size, max_seq_length, emb_dims)
        # Masked mean pooling
        masked_embedded = embedded * mask.unsqueeze(-1)
        sum_embedded = masked_embedded.sum(dim=1)
        valid_counts = mask.sum(dim=1).unsqueeze(-1)
        pooled = sum_embedded / valid_counts.clamp(min=1e-8)
        return pooled

    def forward(self, query_input, pos_answer_input, neg_answer_input):
        """
        Args:
            query_input: Query token indices (batch_size, seq_length)
            pos_answer_input: Positive answer token indices (batch_size, seq_length)
            neg_answer_input: Negative answer token indices (batch_size, seq_length)
        """
        # Get embeddings and mean pool
        query_embedding = self.pad_and_pool(list(query_input))      # (batch_size, emb_dims)
        pos_answer_embedding = self.pad_and_pool(list(pos_answer_input))  # (batch_size, emb_dims)
        neg_answer_embedding = self.pad_and_pool(list(neg_answer_input))  # (batch_size, emb_dims)
        
        # Pass through towers
        query_output = self.query_tower(query_embedding)           # (batch_size, emb_dims)
        pos_answer_output = self.answer_tower(pos_answer_embedding)    # (batch_size, emb_dims)
        neg_answer_output = self.answer_tower(neg_answer_embedding)    # (batch_size, emb_dims)
        
        return query_output, pos_answer_output, neg_answer_output
    

def compute_tower_loss(query_emb, pos_emb, neg_emb, margin=0.3):
    """
    Compute triplet loss with cosine similarity.
    
    Args:
        query_emb: Query embeddings (batch_size, emb_dims)
        pos_emb: Positive answer embeddings (batch_size, emb_dims)
        neg_emb: Negative answer embeddings (batch_size, emb_dims)
        margin: Minimum margin between positive and negative similarities
    
    Returns:
        loss: Scalar loss value
    """
    # Compute similarities
    pos_distance = 1 - F.cosine_similarity(query_emb, pos_emb, dim=1)  # (batch_size,)
    neg_distance = 1 - F.cosine_similarity(query_emb, neg_emb, dim=1)  # (batch_size,)
    
    # Compute triplet loss: max(0, margin - (pos_sim - neg_sim))
    loss = torch.clamp(margin - (neg_distance - pos_distance), min=0.0) 
    
    return loss.mean()