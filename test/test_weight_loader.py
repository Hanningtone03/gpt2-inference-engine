import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from weight_loader import GPT2Weights

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "gpt2-124m")


def test_config_values_match_gpt2_124m():
    weights = GPT2Weights(MODEL_DIR)
    assert weights.n_layer == 12
    assert weights.n_head == 12
    assert weights.n_embd == 768
    assert weights.vocab_size == 50257


def test_token_embedding_shape():
    weights = GPT2Weights(MODEL_DIR)
    assert weights.wte.shape == (50257, 768)


def test_position_embedding_shape():
    weights = GPT2Weights(MODEL_DIR)
    assert weights.wpe.shape == (1024, 768)


def test_all_12_blocks_loaded():
    weights = GPT2Weights(MODEL_DIR)
    assert len(weights.blocks) == 12


def test_block_attention_weight_shapes():
    weights = GPT2Weights(MODEL_DIR)
    block = weights.blocks[0]
    assert block["attn_c_attn_weight"].shape == (768, 2304)
    assert block["attn_c_proj_weight"].shape == (768, 768)


def test_block_mlp_weight_shapes():
    weights = GPT2Weights(MODEL_DIR)
    block = weights.blocks[0]
    assert block["mlp_c_fc_weight"].shape == (768, 3072)
    assert block["mlp_c_proj_weight"].shape == (3072, 768)
