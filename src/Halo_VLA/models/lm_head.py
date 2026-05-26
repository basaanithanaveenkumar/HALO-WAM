import torch.nn as nn


class LMHead(nn.Module):
    """Linear vocabulary projection. Input is expected to be already normalized
    (DecoderTransformer applies the final LayerNorm before returning)."""

    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, final_states):
        return self.lm_head(final_states)