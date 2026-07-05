class KVCache:
    def __init__(self, n_layer):
        self.n_layer = n_layer
        self.k = [None] * n_layer
        self.v = [None] * n_layer
        self.position = 0

    def get(self, layer_idx):
        return self.k[layer_idx], self.v[layer_idx]

    def update(self, layer_idx, k_new, v_new):
        if self.k[layer_idx] is None:
            self.k[layer_idx] = k_new
            self.v[layer_idx] = v_new
        else:
            self.k[layer_idx] = _concat_along_seq(self.k[layer_idx], k_new)
            self.v[layer_idx] = _concat_along_seq(self.v[layer_idx], v_new)
        return self.k[layer_idx], self.v[layer_idx]


def _concat_along_seq(past, new):
    import numpy as np
    return np.concatenate([past, new], axis=1)
