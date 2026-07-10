"""Draft-model configuration (shared across all §6 variants)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DraftConfig:
    # backbone
    hidden_size: int = 2048
    n_layers: int = 4                 # 3-5 (§4.1)
    n_heads: int = 16                 # head_dim = hidden/n_heads = 128
    rope_theta: float = 1e6
    rms_eps: float = 1e-6
    gamma: int = 5                    # draft block size (§7.2)

    # borrowed frozen dims
    vocab_size: int = 248320

    # KV injection (§4.1)
    inject_layers: tuple[int, ...] = (19, 31, 39)

    # MoE draft layers (§4.2)
    router_mode: str = "hard"         # hard | soft | scratch | dense
    K: int = 16                       # draft experts (groups)
    k_prime: int = 2                  # active experts
    moe_intermediate_size: int = 512  # mirror target (or 1024 ablation)
    n_shared: int = 1                 # always-on shared expert(s)
    num_target_experts: int = 256     # source cardinality for C

    # Markov semi-AR head (§4.3)
    markov_rank: int = 256

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.n_heads

    @property
    def n_inject(self) -> int:
        return len(self.inject_layers)

    @property
    def is_moe(self) -> bool:
        return self.router_mode in ("hard", "soft", "scratch")
