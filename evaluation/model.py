"""Connect an experiment GPT to native OLMES likelihood scoring."""

import torch
from oe_eval.models.eleuther_huggingface import HFLM_Verbose

from evaluation.checkpoint import load_checkpoint
from evaluation.count_tokens import BATCH_SIZE, MAX_LENGTH, TokenObserver
from evaluation.tokenizer_adapter import TokenizerAdapter


class CheckpointModel(TokenObserver):
    # Use native scoring, caching, masking, padding and result construction verbatim.
    _loglikelihood_tokens_verbose = HFLM_Verbose._loglikelihood_tokens_verbose

    def __init__(self, checkpoint, tokenizer, device="cuda", attention=None, compile_functions=True):
        super().__init__(TokenizerAdapter(tokenizer, MAX_LENGTH), MAX_LENGTH, BATCH_SIZE)
        self._device = torch.device(device)
        self._model, self.source_sha256 = load_checkpoint(
            checkpoint, device, attention=attention, compile_functions=compile_functions)
        if self._model.config.vocab_size != self.vocab_size:
            raise ValueError("The model and its tokenizer have different vocabulary sizes.")
        if self._model.cos.size(1) < self.max_length:
            raise ValueError("The model cannot support the declared evaluation context.")

    @torch.inference_mode()
    def _model_call(self, inputs, attn_mask=None, labels=None):
        if attn_mask is not None or labels is not None:
            raise ValueError("The experiment is a causal language model.")
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == "cuda"):
            return self._model(inputs)
