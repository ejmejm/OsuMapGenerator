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

def prepare_tensor_seq(seq, max_len, preprocess_text, config, pad=True):
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
    seq_tensors.append(torch.tensor(s[-max_len:], dtype=torch.int64))
    if pad:
      pad_len = max_len - len(seq_tensors[-1])
      seq_tensors[-1] = F.pad(seq_tensors[-1], (0, pad_len), value=PAD_TOKEN)
  seq_tensor = torch.stack(seq_tensors).to(config['device'])
  
  # Sequence dim first
  seq_tensor = seq_tensor.transpose(0, 1)
  seq_mask = gen_seq_mask(seq_tensor.shape[0]).to(config['device'])

  return seq_tensor, seq_mask

def prepare_tensor_seqs(src, tgt, preprocess_text, config):
  """Converts to tensor, pads, and send to device.
  
  Args:
    src: List of strings for transformer encoder
    tgt: List of strings for transformer decoder
    preprocess_text: Function that turns text into a list of numerical tokens
    config: Dict of config parameters
  """
  src_tensor, src_mask = prepare_tensor_seq(
    src, config['max_src_len'], preprocess_text, config)
  tgt_tensor, tgt_mask = prepare_tensor_seq(
    tgt, config['max_tgt_len'], preprocess_text, config)
  return src_tensor, tgt_tensor, src_mask, tgt_mask


def prepare_tensor_vqvae(src, preprocess_text, config):
  """Converts to tensor, pads, and send to device.
  
  Args:
    src: List of strings for transformer encoder
    tgt: List of strings for transformer decoder
    config: Dict of config parameters
  """
  PAD_TOKEN = preprocess_text('<pad>')[0]

  src = [[preprocess_text(obj) for obj in s] for s in src]

  max_src_len = config['max_src_len']


  src_tensors = []
  for s in src:
    one_tensor = []
    for obj in s:
      oneline = torch.tensor(obj[:max_src_len], dtype=torch.float)
      pad_len = max_src_len - len(oneline)
      oneline = F.pad(oneline, (0, pad_len), value=PAD_TOKEN)
      one_tensor.append(oneline)
    
    src_tensors.append(torch.stack(one_tensor))
  src_tensor = torch.stack(src_tensors)

  return src_tensor

def prepare_tensor_vqvae(src, preprocess_text, config):
  """Converts to tensor, pads, and send to device.
  
  Args:
    src: List of strings for transformer encoder
    tgt: List of strings for transformer decoder
    config: Dict of config parameters
  """
  PAD_TOKEN = preprocess_text('<pad>')[0]

  src = [[preprocess_text(obj) for obj in s] for s in src]

  max_src_len = config['max_src_len']


  src_tensors = []
  for s in src:
    one_tensor = []
    for obj in s:
      oneline = torch.tensor(obj[:max_src_len], dtype=torch.float)
      pad_len = max_src_len - len(oneline)
      oneline = F.pad(oneline, (0, pad_len), value=PAD_TOKEN)
      one_tensor.append(oneline)
    
    src_tensors.append(torch.stack(one_tensor))
  src_tensor = torch.stack(src_tensors)

  return src_tensor

# The type of tokenizer depends on the config settings
config = load_config()
if config.get('tokenizer_type') == 'sentencepiece':
  tokenize = None
  raise Exception('Sentencepiece tokenizer not implemented yet')
else:
  tokenize = default_tokenize