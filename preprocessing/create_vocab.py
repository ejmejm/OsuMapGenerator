import os
import sys
from typing import Counter
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torchtext as tt

from utils import load_config, parse_args
from preprocessing.data_loading import get_dataloaders, sample_from_map
from preprocessing.data_loading import format_training_data
from text_processing import get_tokenizer


def create_default_vocab(args, config):
  print('Creating vocab with default tokenizer')
  tokenize = get_tokenizer(config)

  train_loader = get_dataloaders(
    config['beatmap_path'], batch_size=config.get('batch_size'), val_split = config.get('val_split'), test_split = config.get('test_split'))[0]

  token_counts = Counter()
  for batch_idx, batch in enumerate(train_loader):
    batch_samples = [sample_from_map(*map) for map in batch]
    training_samples = [''.join(format_training_data(*map)) for map in batch_samples]
    for sample in training_samples:
      tokens = tokenize(sample)
      token_counts.update(tokens)
    if args.early_stop > 0 and batch_idx >= args.early_stop - 1:
      break

  vocab = tt.vocab.vocab(token_counts, specials=['<unk>', '<pad>'])
  vocab.set_default_index(vocab['<unk>'])
  save_path = os.path.join(config['vocab_dir'], 'default_vocab.pt')
  torch.save(vocab, save_path)

  print(f'Created a vocab with {len(token_counts)} unique tokens')
  print(f'Vocab saved to {save_path}')


def create_sentencepiece_model(args, config):
  print('Creating vocab with sentencepiece')

  train_loader = get_dataloaders(
    config['beatmap_path'], batch_size=config.get('batch_size'), val_split = config.get('val_split'), test_split = config.get('test_split'))[0]

  vocab_size = config['spm_vocab_size']
  tmp_file = 'vocab_train.txt'
  with open(tmp_file, 'w+') as f:
    for batch_idx, batch in enumerate(train_loader):
      batch_samples = [sample_from_map(*map) for map in batch]
      training_samples = [''.join(format_training_data(*map)) for map in batch_samples]
      for sample in training_samples:
        f.write(sample + '\n')
      if args.early_stop > 0 and batch_idx >= args.early_stop - 1:
        break

  save_path = 'preprocessing/vocab/spm'
  tt.data.functional.generate_sp_model(
    tmp_file, vocab_size=vocab_size, model_prefix=save_path,
    model_type='bpe')
  os.remove(tmp_file)

  print(f'Created a vocab with {vocab_size} unique tokens')
  print(f'Vocab saved to {save_path}.model')


if __name__ == '__main__':
  args = parse_args()
  config = load_config(args.config)
  if config.get('tokenizer_type') == 'sentencepiece':
    create_sentencepiece_model(args, config)
  else:
    create_default_vocab(args, config)
    
