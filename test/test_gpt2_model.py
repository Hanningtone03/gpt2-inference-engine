import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gpt2_model import GPT2Model
from weight_loader import GPT2Weights
from bpe_tokenizer import BPETokenizer

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "gpt2-124m")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def make_model():
    weights = GPT2Weights(MODEL_DIR)
    return GPT2Model(weights), weights


def make_tokenizer():
    return BPETokenizer(
        os.path.join(MODELS_DIR, "encoder.json"),
        os.path.join(MODELS_DIR, "vocab.bpe"),
    )


def test_forward_output_shape_matches_vocab_size():
    model, weights = make_model()
    token_ids = [464, 3290]
    logits = model.forward(token_ids)
    assert logits.shape == (2, weights.vocab_size)


def test_forward_produces_finite_values():
    model, _ = make_model()
    token_ids = [464, 3290, 318, 257]
    logits = model.forward(token_ids)
    assert np.all(np.isfinite(logits))


def test_model_predicts_plausible_next_token_for_simple_prompt():
    model, _ = make_model()
    tokenizer = make_tokenizer()

    prompt = "The capital of France is"
    token_ids = tokenizer.encode(prompt)
    logits = model.forward(token_ids)

    next_token_logits = logits[-1]
    top_id = int(np.argmax(next_token_logits))
    predicted = tokenizer.decode([top_id])

    assert predicted.strip().lower() in ("paris", "the", "a")


def test_deterministic_forward_pass_same_input_same_output():
    model, _ = make_model()
    token_ids = [464, 3290, 318]
    logits1 = model.forward(token_ids)
    logits2 = model.forward(token_ids)
    assert np.allclose(logits1, logits2)
