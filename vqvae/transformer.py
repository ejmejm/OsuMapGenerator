import os

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math
from einops import rearrange

def model_from_config(config, vocab):
    # Create the model and load when applicable
    model = VQVaeTransformer(len(vocab) + 1, len(vocab), config['codebook_size'] + 3, config['codebook_size'] + 2, d_src_word_vec=512,
                                        d_trg_word_vec=512,
                                        d_model=config['d_model'], d_inner=config['d_hid'], n_enc_layers=config['n_encoder_layers'],
                                        n_dec_layers=config['n_decoder_layers'], n_head=config['n_head'], d_k=config['d_k'], d_v=config['d_v'],
                                        dropout=config['dropout'],
                                        n_src_position=50, n_trg_position=100,
                                        trg_emb_prj_weight_sharing=False, 
                                         segments_per_beat = config['segments_per_beat'],
                                        audio_tokens_per_segment = config['audio_tokens_per_segment'],
                                        n_mel_bands = config['n_mel_bands'],
                                        include_audio = config['include_audio']
                                        )

    if config['load_model'] and os.path.exists(config['model_save_path']):
        model.load_state_dict(torch.load(config['model_save_path']))

    return model

def get_non_pad_mask(padded_input, input_lengths=None, pad_idx=None):
    """padding position is set to 0, either use input_lengths or pad_idx
    """
    assert input_lengths is not None or pad_idx is not None
    if input_lengths is not None:
        # padded_input: N x T x ..
        N = padded_input.size(0)
        non_pad_mask = padded_input.new_ones(padded_input.size()[:-1])  # N x T
        for i in range(N):
            non_pad_mask[i, input_lengths[i]:] = 0
    if pad_idx is not None:
        # padded_input: N x T
        assert padded_input.dim() == 2
        non_pad_mask = padded_input.ne(pad_idx).float()
    # unsqueeze(-1) for broadcast
    return non_pad_mask.unsqueeze(-1)

def get_pad_mask(batch_size, seq_len, non_pad_lens):
    non_pad_lens = non_pad_lens.data.tolist()
    mask_2d = torch.zeros((batch_size, seq_len), dtype=torch.float32)
    for i, cap_len in enumerate(non_pad_lens):
        mask_2d[i, :cap_len] = 1
    return mask_2d.unsqueeze(1).bool()

def get_pad_mask_idx(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(1)

def get_subsequent_mask(seq):
    sz_b, seq_len = seq.shape
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, seq_len, seq_len)), diagonal=1)).bool()
    return subsequent_mask.to(seq.device)

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        self.fc = nn.Linear(n_head * d_v, d_model)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output,
                slf_attn_mask=None, dec_enc_attn_mask=None):

        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input,
                                                 mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output,
                                                 mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn

class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class Encoder(nn.Module):
    def __init__(self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner,
                 pad_idx, dropout=0.1, n_position=40):
        super(Encoder, self).__init__()
        self.position_enc = PositionalEncoding(d_model)
        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        # self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False, input_onehot=False):
        enc_slf_attn_list = []
        # if input_onehot:
        #     src_seq = torch.matmul(src_seq, self.src_word_emb.weight)
        #     src_seq = src_seq * src_mask.transpose(1, 2)
        # else:
        #     src_seq = self.src_word_emb(src_seq)
        # src_seq *= self.d_model ** 0.5
        enc_output = self.position_enc(src_seq)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class Decoder(nn.Module):
    def __init__(self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
                 d_model, d_inner, pad_idx, n_position=200, dropout=0.1):
        super(Decoder, self).__init__()
        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):
        dec_slf_attn_list, dec_enc_attn_list = [], []

        dec_output = self.trg_word_emb(trg_seq)
        dec_output *= self.d_model ** 0.5

        dec_output = self.position_enc(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,

class VQVaeTransformer(nn.Module):
    def __init__(self, n_src_vocab, src_pad_idx, n_trg_vocab, trg_pad_idx, d_src_word_vec=512, d_trg_word_vec=512,
                 d_model=512, d_inner=2048, n_enc_layers=6, n_dec_layers=6, n_head=8, d_k=64, d_v=64,
                 dropout=0.1, n_src_position=40, n_trg_position=200, trg_emb_prj_weight_sharing=True, segments_per_beat: int = 8, audio_tokens_per_segment: int = 2,
                 n_mel_bands: int = 80, include_audio=False):
        super(VQVaeTransformer, self).__init__()
        self.include_audio = include_audio
        self.trg_pad_idx = trg_pad_idx
        self.src_pad_idx = src_pad_idx

        self.d_model = d_model

        if self.include_audio:
            self.audio_layers = nn.Sequential(
                nn.Conv1d(n_mel_bands, n_mel_bands, kernel_size=7, stride=3),
                nn.ReLU(),
                nn.Conv1d(n_mel_bands, n_mel_bands, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv1d(n_mel_bands, n_mel_bands, kernel_size=4),
                nn.ReLU(),
                nn.Conv1d(n_mel_bands, self.d_model, kernel_size=3),
                nn.ReLU(),
                nn.Dropout1d(dropout),
                nn.AdaptiveAvgPool1d(audio_tokens_per_segment)
            )

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_src_position, d_word_vec=d_src_word_vec,
            d_model=d_model, d_inner=d_inner, n_layers=n_enc_layers, n_head=n_head, d_k=d_k,
            d_v=d_v,  pad_idx=src_pad_idx, dropout=dropout
        )

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_trg_position, d_word_vec=d_trg_word_vec,
            d_model=d_model, d_inner=d_inner, n_layers=n_dec_layers, n_head=n_head, d_k=d_k,
            d_v=d_v, pad_idx=trg_pad_idx, dropout=dropout
        )
        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)
        for p in self.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)

        if trg_emb_prj_weight_sharing:
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

    def forward(self, src_seq, trg_seq, audio = None, input_onehot=False, src_mask=None, src_non_pad_lens=0):
        batch_size, src_seq_len = src_seq.shape[0], src_seq.shape[1]
        # src_mask = get_pad_mask(batch_size, src_seq_len, src_non_pad_lens).to(src_seq.device)
        if not input_onehot:
            src_mask = get_pad_mask_idx(src_seq, self.src_pad_idx)
        elif src_mask is None:
            src_mask = get_pad_mask(batch_size, src_seq_len, src_non_pad_lens).to(src_seq.device)
        trg_mask = get_pad_mask_idx(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        # print(src_mask)
        # print(trg_mask)

        if self.include_audio and audio is not None:
            audio_in_shape = audio.shape
            # audio = rearrange(audio, 'b s c d -> (b s) c d')
            audio_embeds = self.audio_layers(audio)
            # audio_embeds = rearrange(audio_embeds,
                # '(b s) c d -> b (s d) c', b=audio_in_shape[0], s=audio_in_shape[1])
        audio_mask = get_non_pad_mask(audio_embeds, [ audio_embeds.shape[1] for i in range(audio_embeds.shape[0])])
        # enc_output, *_ = self.encoder(src_seq, src_mask, input_onehot, input_onehot=input_onehot)
        enc_output, *_ = self.encoder(audio_embeds, audio_mask, input_onehot)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        seq_logit = self.trg_word_prj(dec_output)
        return seq_logit

    def sample(self, src_seq, trg_sos, trg_eos, max_steps=100, sample=False, top_k=None):
        trg_seq = torch.LongTensor(src_seq.size(0), 1).fill_(trg_sos).to(src_seq).long()

        # batch_size, src_seq_len = src_seq.shape[0], src_seq.shape[1]
        # src_mask = get_pad_mask(batch_size, src_seq_len, src_non_pad_lens).to(src_seq.device)
        src_mask = get_pad_mask_idx(src_seq, self.src_pad_idx)
        enc_output, *_ = self.encoder(src_seq, src_mask)

        for _ in range(max_steps):
            # print(trg_seq)
            trg_mask = get_subsequent_mask(trg_seq)
            dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
            seq_logit = self.trg_word_prj(dec_output)
            logits = seq_logit[:, -1, :]

            if top_k is not None:
                logits = top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            _, ix = torch.topk(probs, k=1, dim=-1)
            if ix[0] == trg_eos:
                break

            if sample:
                ix = torch.multinomial(probs, num_samples=1)
                while (ix[0] in [trg_sos, trg_eos]):
                    ix = torch.multinomial(probs, num_samples=1)
            trg_seq = torch.cat((trg_seq, ix), dim=1)
        return trg_seq

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

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
    def __init__(self, d_model, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        self.register_buffer('positional_encoding', self.encoding)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        return self.encoding[:seq_len, :].clone().detach().to(x.device) + x