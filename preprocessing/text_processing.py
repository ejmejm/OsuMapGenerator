import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from torch.nn import functional as F
import torchtext as tt

from models import gen_seq_mask


def default_tokenize(string):
  """Default tokenizer splits on "<>" tags and individual characters."""
  tokens = ['']
  in_tag = False
  for char in string:
    if char == '<':
      in_tag = True
      if tokens[-1] == '':
        tokens[-1] += char
      else:
        tokens.append(char)
    elif char == '>':
      in_tag = False
      tokens[-1] += char
    else:
      if in_tag:
        tokens[-1] += char
      else:
        tokens.append(char)
  return tokens

class StandardNumericalizer():
  def __init__(self, vocab):
    self.vocab = vocab

  def __call__(self, string):
    tokens = default_tokenize(string)
    return self.vocab(tokens)

def get_text_preprocessor(config):
  """Returns a function that turns text into a list of numerical tokens."""
  if config['tokenizer_type'] == 'sentencepiece':
    spm_path = os.path.join(config['vocab_dir'], 'spm.model')
    spm = tt.data.functional.load_sp_model(spm_path)
    spn = tt.data.functional.sentencepiece_numericalizer(spm)
    return spn, spm
  else:
    vocab_path = os.path.join(config['vocab_dir'], 'default_vocab.pt')
    # Load file with torch
    vocab = torch.load(vocab_path)

    return StandardNumericalizer(vocab), vocab

def prepare_tensor_seq(seq, max_len, preprocess_text, config, pad=True, device=None):
  """Converts to tensor, pads, and send to device.
  
  Args:
    src: List of strings for transformer encoder
    preprocess_text: Function that turns text into a list of numerical tokens
    config: Dict of config parameters
  """
  PAD_TOKEN = preprocess_text('<pad>')[0]

  seq = [preprocess_text(s) for s in seq]

  seq_tensors = []
  for s in seq:
    seq_tensors.append(torch.tensor(s[:max_len], dtype=torch.int64))
    if pad:
      pad_len = max_len - len(seq_tensors[-1])
      seq_tensors[-1] = F.pad(seq_tensors[-1], (0, pad_len), value=PAD_TOKEN)
  seq_tensor = torch.stack(seq_tensors).to(device or config['device'])
  
  # Sequence dim first
  seq_tensor = seq_tensor.transpose(0, 1)
  seq_mask = gen_seq_mask(seq_tensor.shape[0]).to(device or config['device'])

  return seq_tensor, seq_mask

def prepare_tensor_seqs(src, tgt, preprocess_text, config, device=None):
  """Converts to tensor, pads, and send to device.
  
  Args:
    src: List of strings for transformer encoder
    tgt: List of strings for transformer decoder
    preprocess_text: Function that turns text into a list of numerical tokens
    config: Dict of config parameters
  """
  src_tensor, src_mask = prepare_tensor_seq(
    src, config['max_src_len'], preprocess_text, config, device=device)
  tgt_tensor, tgt_mask = prepare_tensor_seq(
    tgt, config['max_tgt_len'], preprocess_text, config, device=device)
  return src_tensor, tgt_tensor, src_mask, tgt_mask

# The type of tokenizer depends on the config settings
def get_tokenizer(config):
  if config.get('tokenizer_type') == 'sentencepiece':
    tokenize = None
    raise Exception('Sentencepiece tokenizer not implemented yet')
  else:
    tokenize = default_tokenize
  return tokenize