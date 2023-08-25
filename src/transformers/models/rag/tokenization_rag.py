# coding=utf-8
# Copyright 2020, The RAG Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes for RAG."""
import os
import warnings
from typing import List, Optional

from ...tokenization_utils_base import BatchEncoding
from ...utils import logging
from .configuration_rag import RagConfig
from ...tokenization_utils import PreTrainedTokenizer  #elephant

logger = logging.get_logger(__name__)


class RagTokenizer(PreTrainedTokenizer): #elephant
    def __init__(self, question_encoder, generator):
        self.question_encoder = question_encoder
        self.generator = generator
        self.current_tokenizer = self.question_encoder
        #elephant
        unk_token="[UNK]"
        pad_token="[PAD]"
        extra_ids=100
        additional_special_tokens=None
        
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            #sp_model_kwargs=self.sp_model_kwargs,
            #**kwargs,
        )
        
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        with open("vocab.txt", encoding="utf-8") as vocab_handle:
            self.encoder=vocab_handle.readlines()
        for i in range(len(self.encoder)):
            self.encoder[i]=self.encoder[i].strip()
        if token in self.encoder:
            return self.encoder.index(token)
        return self.encoder.index(self.unk_token) 

    def save_pretrained(self, save_directory):
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)
        question_encoder_path = os.path.join(save_directory, "question_encoder_tokenizer")
        generator_path = os.path.join(save_directory, "generator_tokenizer")
        self.question_encoder.save_pretrained(question_encoder_path)
        self.generator.save_pretrained(generator_path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # dynamically import AutoTokenizer
        from ..auto.tokenization_auto import AutoTokenizer

        config = kwargs.pop("config", None)

        if config is None:
            config = RagConfig.from_pretrained(pretrained_model_name_or_path)

        question_encoder = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, config=config.question_encoder, subfolder="question_encoder_tokenizer"
        )
        generator = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, config=config.generator, subfolder="generator_tokenizer"
        )
        return cls(question_encoder=question_encoder, generator=generator)

    def __call__(self, *args, **kwargs):
        return self.current_tokenizer(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.generator.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.generator.decode(*args, **kwargs)

    def _switch_to_input_mode(self):
        self.current_tokenizer = self.question_encoder

    def _switch_to_target_mode(self):
        self.current_tokenizer = self.generator

    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        tgt_texts: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        padding: str = False, #elephant
        return_tensors: str = None,
        truncation: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        warnings.warn(
            "`prepare_seq2seq_batch` is deprecated and will be removed in version 5 of 🤗 Transformers. Use the "
            "regular `__call__` method to prepare your inputs and the tokenizer under the `with_target_tokenizer` "
            "context manager to prepare your targets. See the documentation of your specific tokenizer for more "
            "details",
            FutureWarning,
        )
        if max_length is None:
            max_length = self.current_tokenizer.model_max_length
         if src_texts is not None:  #elephant
            model_inputs = self(
                src_texts,
                add_special_tokens=True,
                return_tensors=return_tensors,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                **kwargs,
            )
        if tgt_texts is None:
            return model_inputs
        # Process tgt_texts
        if max_target_length is None:
            max_target_length = self.current_tokenizer.model_max_length
        labels = self(
            text_target=tgt_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            padding=padding,
            max_length=max_target_length,
            truncation=truncation,
            **kwargs,
        )
        #elephant
        return labels #elephant
