import numpy as np
from ops import layer_norm
from transformer_block import transformer_block


class GPT2Model:
    def __init__(self, weights):
        self.weights = weights

    def forward(self, token_ids):
        seq_len = len(token_ids)
        token_embeddings = self.weights.wte[token_ids]
        position_embeddings = self.weights.wpe[:seq_len]

        x = token_embeddings + position_embeddings

        for block_weights in self.weights.blocks:
            x = transformer_block(x, block_weights, self.weights.n_head)

        x = layer_norm(x, self.weights.ln_f_weight, self.weights.ln_f_bias)

        logits = x @ self.weights.wte.T
        return logits
