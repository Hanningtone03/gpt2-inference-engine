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


class GPT2ModelCached(GPT2Model):
    def forward_with_cache(self, new_token_ids, cache):
        from transformer_block import transformer_block_with_cache

        position_offset = cache.position
        new_len = len(new_token_ids)

        token_embeddings = self.weights.wte[new_token_ids]
        position_embeddings = self.weights.wpe[position_offset:position_offset + new_len]
        x = token_embeddings + position_embeddings

        for layer_idx, block_weights in enumerate(self.weights.blocks):
            x = transformer_block_with_cache(x, block_weights, self.weights.n_head, cache, layer_idx, position_offset)

        cache.position += new_len

        x = layer_norm(x, self.weights.ln_f_weight, self.weights.ln_f_bias)
        logits = x @ self.weights.wte.T
        return logits


class GPT2ModelCachedWithAttention(GPT2ModelCached):
    def forward_with_cache_and_attention(self, new_token_ids, cache):
        from transformer_block import transformer_block_with_cache_and_weights

        position_offset = cache.position
        new_len = len(new_token_ids)

        token_embeddings = self.weights.wte[new_token_ids]
        position_embeddings = self.weights.wpe[position_offset:position_offset + new_len]
        x = token_embeddings + position_embeddings

        last_layer_weights = None
        for layer_idx, block_weights in enumerate(self.weights.blocks):
            x, attn_weights = transformer_block_with_cache_and_weights(
                x, block_weights, self.weights.n_head, cache, layer_idx, position_offset
            )
            last_layer_weights = attn_weights

        cache.position += new_len

        x = layer_norm(x, self.weights.ln_f_weight, self.weights.ln_f_bias)
        logits = x @ self.weights.wte.T
        return logits, last_layer_weights
