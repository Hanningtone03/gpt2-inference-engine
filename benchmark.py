import time
import sys
sys.path.insert(0, 'src')

from generate import generate, generate_cached
from gpt2_model import GPT2Model, GPT2ModelCached
from weight_loader import GPT2Weights
from bpe_tokenizer import BPETokenizer

weights = GPT2Weights('models/gpt2-124m')
plain_model = GPT2Model(weights)
cached_model = GPT2ModelCached(weights)
tokenizer = BPETokenizer('models/encoder.json', 'models/vocab.bpe')

prompt = "The history of computing began"
n_tokens = 30

start = time.time()
uncached_result = generate(plain_model, tokenizer, prompt, max_new_tokens=n_tokens, strategy="greedy")
uncached_time = time.time() - start

start = time.time()
cached_result = generate_cached(cached_model, tokenizer, prompt, max_new_tokens=n_tokens, strategy="greedy")
cached_time = time.time() - start

print(f"Uncached: {uncached_time:.2f}s ({n_tokens / uncached_time:.2f} tokens/sec)")
print(f"Cached:   {cached_time:.2f}s ({n_tokens / cached_time:.2f} tokens/sec)")
print(f"Speedup:  {uncached_time / cached_time:.2f}x")
print(f"Outputs match: {uncached_result == cached_result}")
