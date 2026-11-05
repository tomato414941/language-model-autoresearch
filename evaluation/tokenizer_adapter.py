"""Expose the experiment's existing tokenizer to the native OLMES model wrapper."""

from bisect import bisect_left, bisect_right
import pickle


class TokenizerAdapter:
    def __init__(self, path, max_length=20480):
        with open(path, "rb") as source:
            self.encoding = pickle.load(source)
        self.bos_token_id = self.encoding.encode_single_token("<|reserved_0|>")
        self.eos_token_id = self.bos_token_id
        self.pad_token_id = self.bos_token_id
        self.model_max_length = max_length
        self.padding_side = "right"
        self.name_or_path = "autoresearch-tokenizer"
        self.chat_template = None

    def __len__(self):
        return self.encoding.n_vocab

    def encode(self, text, add_special_tokens=False, **kwargs):
        tokens = self.encoding.encode_ordinary(text)
        return [self.bos_token_id] + tokens if add_special_tokens else tokens

    def decode(self, tokens, skip_special_tokens=False, **kwargs):
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        if skip_special_tokens:
            tokens = [token for token in tokens if token < self.bos_token_id]
        return self.encoding.decode(tokens)

    def encode_plus(self, text, return_offsets_mapping=False, add_special_tokens=False, **kwargs):
        tokens = self.encode(text, add_special_tokens=False)
        result = {"input_ids": tokens, "attention_mask": [1] * len(tokens)}
        if return_offsets_mapping:
            boundaries = [0]
            for character in text:
                boundaries.append(boundaries[-1] + len(character.encode("utf-8")))
            offsets = []
            byte_offset = 0
            for token in tokens:
                next_offset = byte_offset + len(self.encoding.decode_single_token_bytes(token))
                offsets.append((bisect_right(boundaries, byte_offset) - 1,
                                bisect_left(boundaries, next_offset)))
                byte_offset = next_offset
            assert byte_offset == boundaries[-1]
            result["offset_mapping"] = offsets
        if add_special_tokens:
            result["input_ids"] = [self.bos_token_id] + tokens
            result["attention_mask"] = [1] + result["attention_mask"]
            if return_offsets_mapping:
                result["offset_mapping"] = [(0, 0)] + result["offset_mapping"]
        return result
