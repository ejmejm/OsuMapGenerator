import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.nn import functional as F
import torchtext as tt

from models import gen_seq_mask
from utils import load_config

def prepare_tensor_vqvae(src, preprocess_text, config):
  """Converts to tensor, pads, and send to device.
  
  Args:
    src: List of strings for transformer encoder
    tgt: List of strings for transformer decoder
    config: Dict of config parameters
  """
  # only take x, y, time, type, hitSound into consideration now
  # after the first version, we can add other attributes
  def process(hitobject):
    s_arr = hitobject.split(',')
    return [int(s_arr[0]), int(s_arr[1]), int(s_arr[2]), int(s_arr[3]), int(s_arr[4])]

  src = [[process(obj) for obj in s] for s in src]

  max_src_len = config['max_src_len']

  PAD_TOKEN = 9999999

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

def prepare_tensor_transformer(meta, audio, tokens, preprocess_text, config):
  """Converts to tensor, pads, and send to device.
  
  Args:
    src: List of strings for transformer encoder
    tgt: List of strings for transformer decoder
    preprocess_text: Function that turns text into a list of numerical tokens
    config: Dict of config parameters
  """
  src_tensor, src_mask = prepare_tensor_input(
    meta, config['max_src_len'], config['max_src_len'], preprocess_text, config)
  
  token_tensor = torch.tensor(tokens)
  tgt_mask = gen_seq_mask(len(tokens[0])).to(config['device'])
  
  return src_tensor, token_tensor, src_mask, tgt_mask

def prepare_tensor_input(meta, audio, max_len, preprocess_text, config, pad=True):
  """Converts to tensor, pads, and send to device.
  
  Args:
    src: List of strings for transformer encoder
    tgt: List of strings for transformer decoder
    config: Dict of config parameters
  """
  PAD_TOKEN = preprocess_text('<pad>')[0]

  seq = [preprocess_text(s) for s in meta]

  seq_tensors = []
  for s in seq:
    seq_tensors.append(torch.tensor(s[-max_len:], dtype=torch.int64))
    if pad:
      pad_len = max_len - len(seq_tensors[-1])
      seq_tensors[-1] = F.pad(seq_tensors[-1], (0, pad_len), value=PAD_TOKEN)
  seq_tensor = torch.stack(seq_tensors).to(config['device'])
  
  # # Sequence dim first
  # seq_tensor = seq_tensor.transpose(0, 1)
  seq_mask = gen_seq_mask(seq_tensor.shape[1]).to(config['device'])

  return seq_tensor, seq_mask