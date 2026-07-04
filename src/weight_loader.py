import json
from safetensors import safe_open


class GPT2Weights:
    def __init__(self, model_dir):
        with open(f"{model_dir}/config.json", "r") as f:
            self.config = json.load(f)

        self.n_layer = self.config["n_layer"]
        self.n_head = self.config["n_head"]
        self.n_embd = self.config["n_embd"]
        self.vocab_size = self.config["vocab_size"]
        self.n_ctx = self.config["n_ctx"]

        self.raw = {}
        with safe_open(f"{model_dir}/model.safetensors", framework="numpy") as f:
            for key in f.keys():
                self.raw[key] = f.get_tensor(key)

        self.wte = self.raw["wte.weight"]
        self.wpe = self.raw["wpe.weight"]
        self.ln_f_weight = self.raw["ln_f.weight"]
        self.ln_f_bias = self.raw["ln_f.bias"]

        self.blocks = []
        for i in range(self.n_layer):
            prefix = f"h.{i}."
            block = {
                "ln_1_weight": self.raw[prefix + "ln_1.weight"],
                "ln_1_bias": self.raw[prefix + "ln_1.bias"],
                "attn_c_attn_weight": self.raw[prefix + "attn.c_attn.weight"],
                "attn_c_attn_bias": self.raw[prefix + "attn.c_attn.bias"],
                "attn_c_proj_weight": self.raw[prefix + "attn.c_proj.weight"],
                "attn_c_proj_bias": self.raw[prefix + "attn.c_proj.bias"],
                "ln_2_weight": self.raw[prefix + "ln_2.weight"],
                "ln_2_bias": self.raw[prefix + "ln_2.bias"],
                "mlp_c_fc_weight": self.raw[prefix + "mlp.c_fc.weight"],
                "mlp_c_fc_bias": self.raw[prefix + "mlp.c_fc.bias"],
                "mlp_c_proj_weight": self.raw[prefix + "mlp.c_proj.weight"],
                "mlp_c_proj_bias": self.raw[prefix + "mlp.c_proj.bias"],
            }
            self.blocks.append(block)
