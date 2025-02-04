from torch import nn
import torch

# Build Word2Vec model
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, emb_dims):
        super().__init__()
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(num_embeddings= vocab_size, embedding_dim= emb_dims)
        self.output_weights = nn.Linear(in_features = emb_dims, out_features = vocab_size)
        self.sigmoid = nn.Sigmoid()
        self.loss_history = []

    def forward(self, x, positive_samples, negative_samples):
        # Add dimension checks
        if x.max() >= self.vocab_size or positive_samples.max() >= self.vocab_size or negative_samples.max() >= self.vocab_size:
            raise ValueError(f"Input indices must be less than vocab_size ({self.vocab_size})")

        emb = self.embeddings(x)
        context_weights = self.output_weights.weight[positive_samples]
        negative_sample_weights = self.output_weights.weight[negative_samples]
        positive_out = torch.bmm(context_weights, emb.unsqueeze(-1)).squeeze(-1)
        negative_out = torch.bmm(negative_sample_weights, emb.unsqueeze(-1)).squeeze(-1)
        positive_out = self.sigmoid(positive_out)
        negative_out = self.sigmoid(negative_out)
        positive_loss = -positive_out.log().mean()
        negative_loss = -(1 - negative_out + 10**(-3)).log().mean()
        return positive_loss + negative_loss