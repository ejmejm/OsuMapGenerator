import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.nn import functional as F
import torchtext as tt

from models import gen_seq_mask
from utils import load_config


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

    def numericalize(string):
      tokens = default_tokenize(string)
      return vocab(tokens)

    return numericalize, vocab

def prepare_tensor_seqs(src, tgt, preprocess_text, config):
  """Converts to tensor, pads, and send to device.
  
  Args:
    src: List of strings for transformer encoder
    tgt: List of strings for transformer decoder
    config: Dict of config parameters
  """
  PAD_TOKEN = preprocess_text('<pad>')[0]

  src = [preprocess_text(s) for s in src]
  tgt = [preprocess_text(t) for t in tgt]

  max_src_len = config['max_src_len']
  max_tgt_len = config['max_tgt_len']

  src_tensors = []
  tgt_tensors = []
  for s, t in zip(src, tgt):
    src_tensors.append(torch.tensor(s[:max_src_len], dtype=torch.int64))
    pad_len = max_src_len - len(src_tensors[-1])
    src_tensors[-1] = F.pad(src_tensors[-1], (0, pad_len), value=PAD_TOKEN)

    tgt_tensors.append(torch.tensor(t[:max_tgt_len], dtype=torch.int64))
    pad_len = max_tgt_len - len(tgt_tensors[-1])
    tgt_tensors[-1] = F.pad(tgt_tensors[-1], (0, pad_len), value=PAD_TOKEN)

  src_tensor = torch.stack(src_tensors).to(config['device'])
  tgt_tensor = torch.stack(tgt_tensors).to(config['device'])

  # Sequence dim first
  src_tensor = src_tensor.transpose(0, 1)
  tgt_tensor = tgt_tensor.transpose(0, 1)

  src_mask = gen_seq_mask(src_tensor.shape[0]).to(config['device'])
  tgt_mask = gen_seq_mask(tgt_tensor.shape[0]).to(config['device'])

  return src_tensor, tgt_tensor, src_mask, tgt_mask


# The type of tokenizer depends on the config settings
config = load_config()
if config.get('tokenizer_type') == 'sentencepiece':
  tokenize = None
  raise Exception('Sentencepiece tokenizer not implemented yet')
else:
  tokenize = default_tokenize