import math
import os

from einops import rearrange
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F


def model_from_config(config, vocab):
    # Create the model and load when applicable
    model = DefaultTransformer(
        n_token = len(vocab),
        d_model = config['d_model'],
        n_head = config['n_head'],
        d_hid = config['d_hid'],
        n_encoder_layers = config['n_encoder_layers'],
        n_decoder_layers = config['n_decoder_layers'],
        dropout = config['dropout'],
        segments_per_beat = config['segments_per_beat'],
        audio_tokens_per_segment = config['audio_tokens_per_segment'],
        n_mel_bands = config['n_mel_bands'],
        include_audio = config['include_audio'],
    ).to(config['device'])

    if config['load_model'] and os.path.exists(config['model_save_path']):
        model.load_state_dict(torch.load(config['model_save_path']))

    return model

class DefaultTransformer(nn.Module):
    def __init__(self, n_token: int, d_model: int, n_head: int, d_hid: int,
                 n_encoder_layers: int, n_decoder_layers: int, dropout: float = 0.5,
                 segments_per_beat: int = 8, audio_tokens_per_segment: int = 2,
                 n_mel_bands: int = 80, include_audio=False):
        super().__init__()

        self.include_audio = include_audio
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model, n_head, n_encoder_layers, n_decoder_layers, d_hid, dropout)
        self.embedding = nn.Embedding(n_token, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, n_token)

        if self.include_audio:
            self.audio_layers = nn.Sequential(
                nn.Conv1d(n_mel_bands, n_mel_bands, kernel_size=5, stride=3),
                nn.ReLU(),
                nn.Conv1d(n_mel_bands, n_mel_bands, kernel_size=5, stride=3),
                nn.ReLU(),
                nn.Conv1d(n_mel_bands, self.d_model, kernel_size=3),
                nn.ReLU(),
                nn.Dropout1d(dropout),
                nn.AdaptiveAvgPool1d(audio_tokens_per_segment)
            )

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, tgt: Tensor,
                src_mask: Tensor = None, tgt_mask: Tensor = None,
                audio: Tensor = None, audio_mask: Tensor = None) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]
        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """

        # TODO: extract song as numpy array from src and use here + adjust the 16 as necessary
        if self.include_audio and audio is not None:
            audio_in_shape = audio.shape
            audio = rearrange(audio, 'b s c d -> (b s) c d')
            audio_embeds = self.audio_layers(audio)
            audio_embeds = rearrange(audio_embeds,
                '(b s) c d -> b (s d) c', b=audio_in_shape[0], s=audio_in_shape[1])

        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)

        # Insert audio embeddings into the the target sequence
        audio_idxs = [mask.nonzero() for mask in audio_mask.transpose(0, 1)]
        n_audio_tokens = [len(idxs) for idxs in audio_idxs]
        for batch_idx in range(len(n_audio_tokens)):
            audio_token_idxs = torch.tensor(range(n_audio_tokens[batch_idx]))
            audio_token_idxs = audio_token_idxs[:, None] \
                .repeat(1, self.d_model).to(audio_embeds.device)
            segments = audio_embeds[batch_idx, :n_audio_tokens[batch_idx]]

            scatter_idxs = audio_idxs[batch_idx].repeat(1, self.d_model)
            tgt[:, batch_idx].scatter_(0, scatter_idxs, segments)
        
        src *= math.sqrt(self.d_model)
        tgt *= math.sqrt(self.d_model)

        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        
        output = self.transformer(src, tgt, src_mask, tgt_mask)
        output = self.decoder(output)
        
        return output

def gen_seq_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

######################################################################
# ``PositionalEncoding`` module injects some information about the
# relative or absolute position of the tokens in the sequence. The
# positional encodings have the same dimension as the embeddings so that
# the two can be summed. Here, we use ``sine`` and ``cosine`` functions of
# different frequencies.
#

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)