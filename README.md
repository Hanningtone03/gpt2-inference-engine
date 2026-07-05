![CI](https://github.com/Hanningtone03/gpt2-inference-engine/actions/workflows/ci.yml/badge.svg)

# gpt2-inference-engine

GPT-2 inference implemented in NumPy. Pretrained GPT-2 124M weights, hand-written attention, layer norm, and sampling.

## How it works

The tokenizer replicates GPT-2's exact byte-level BPE scheme: every byte is remapped to a printable Unicode character via a fixed lookup table, then BPE merges are applied on top of that remapped text, so arbitrary binary-safe input never produces an unknown token.

The model itself loads real GPT-2 124M weights from HuggingFace's safetensors format directly into NumPy arrays, with no framework touching the weights at any point. The forward pass implements every core transformer operation by hand: multi-head causal self-attention with explicit query/key/value splitting and causal masking, layer normalization, GELU activation, and residual connections across all 12 transformer blocks, ending in a projection back onto the vocabulary via the tied token embedding matrix.

Generation supports greedy, temperature, top-k, and nucleus (top-p) sampling. A KV cache avoids recomputing attention over the full sequence at every generation step — verified to produce bit-identical output to full recomputation, just faster.

## Verified correctness

- Cached and uncached generation produce byte-identical output text
- Causal masking verified to prevent any information leakage from future tokens
- Full pipeline produces coherent, grammatically correct GPT-2-quality text for real prompts

## Benchmark

30-token generation, GPT-2 124M, CPU:

    Uncached: 50.33s (0.60 tokens/sec)
    Cached:   34.08s (0.88 tokens/sec)
    Speedup:  1.48x

## Project structure

    src/
    ├── byte_encoder.py       GPT-2's byte-to-unicode remapping
    ├── bpe_tokenizer.py      byte-level BPE tokenizer
    ├── weight_loader.py      loads real GPT-2 weights into NumPy
    ├── ops.py                layer norm, GELU, softmax
    ├── attention.py          multi-head causal self-attention
    ├── transformer_block.py  full transformer block (attention + MLP + residuals)
    ├── gpt2_model.py         full forward pass, plain and KV-cached
    ├── kv_cache.py           key/value cache for efficient generation
    ├── sampling.py           greedy, temperature, top-k, top-p sampling
    └── generate.py           autoregressive generation loop

## Running

    python3 src/download_weights.py
    python3 -c "
    import sys; sys.path.insert(0, 'src')
    from generate import generate
    from gpt2_model import GPT2Model
    from weight_loader import GPT2Weights
    from bpe_tokenizer import BPETokenizer

    weights = GPT2Weights('models/gpt2-124m')
    model = GPT2Model(weights)
    tokenizer = BPETokenizer('models/encoder.json', 'models/vocab.bpe')
    print(generate(model, tokenizer, 'Your prompt here', max_new_tokens=40, strategy='top_k', seed=42))
    "

## Testing

    python3 -m pytest test/ -v

## Tech

- Python, NumPy only
- Real GPT-2 124M weights (OpenAI/HuggingFace)
- No PyTorch, TensorFlow, or any ML framework

## License

MIT
