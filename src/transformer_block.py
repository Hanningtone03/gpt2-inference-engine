import numpy as np
from ops import layer_norm, gelu
from attention import causal_self_attention


def mlp(x, c_fc_weight, c_fc_bias, c_proj_weight, c_proj_bias):
    hidden = gelu(x @ c_fc_weight + c_fc_bias)
    return hidden @ c_proj_weight + c_proj_bias


def transformer_block(x, block_weights, n_head):
    ln1_out = layer_norm(x, block_weights["ln_1_weight"], block_weights["ln_1_bias"])
    attn_out = causal_self_attention(
        ln1_out,
        block_weights["attn_c_attn_weight"],
        block_weights["attn_c_attn_bias"],
        block_weights["attn_c_proj_weight"],
        block_weights["attn_c_proj_bias"],
        n_head,
    )
    x = x + attn_out

    ln2_out = layer_norm(x, block_weights["ln_2_weight"], block_weights["ln_2_bias"])
    mlp_out = mlp(
        ln2_out,
        block_weights["mlp_c_fc_weight"],
        block_weights["mlp_c_fc_bias"],
        block_weights["mlp_c_proj_weight"],
        block_weights["mlp_c_proj_bias"],
    )
    x = x + mlp_out

    return x
