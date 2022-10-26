import argparse
import os

from einops import rearrange
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from models import DefaultTransformer
from preprocessing.data_loading import get_dataloaders, sample_from_map
from preprocessing.data_loading import format_training_data
from preprocessing.text_processing import get_text_preprocessor, prepare_tensor_seqs
from utils import load_config

# Create arguments
parser = argparse.ArgumentParser()
# parser.add_argument('--batch_size', type=int, default=32)


BEATMAP_PATH = 'data/formatted_beatmaps/'


def eval(model, data_loader, preprocess_text, config):
  losses = []
  for batch in tqdm(data_loader):
    batch_samples = [sample_from_map(*map) for map in batch]
    training_samples = [format_training_data(*map) for map in batch_samples]

    src, tgt = zip(*training_samples)
    src_tensor, tgt_tensor, src_mask, tgt_mask = prepare_tensor_seqs(src, tgt, preprocess_text, config)
    target = tgt_tensor[1:]
    tgt_tensor = tgt_tensor[:-1]
    tgt_mask = tgt_mask[:-1, :-1]

    with torch.no_grad():
      output = model(src_tensor, tgt_tensor, src_mask, tgt_mask)
    output = rearrange(output, 's b d -> b d s')
    target = rearrange(target, 's b -> b s')

    with torch.no_grad():
      loss = F.cross_entropy(output, target)
    losses.append(loss.item())

  return losses
  

def train(model, train_loader, optimizer, preprocess_text, config, val_loader=None):
  last_eval = 0
  curr_idx = 0

  losses = []
  for epoch_idx in range(config['epochs']):
    for batch in (pbar := tqdm(train_loader)):
      batch_samples = [sample_from_map(*map) for map in batch]
      training_samples = [format_training_data(*map) for map in batch_samples]

      src, tgt = zip(*training_samples)
      src_tensor, tgt_tensor, src_mask, tgt_mask = prepare_tensor_seqs(src, tgt, preprocess_text, config)
      target = tgt_tensor[1:]
      tgt_tensor = tgt_tensor[:-1]
      tgt_mask = tgt_mask[:-1, :-1]

      output = model(src_tensor, tgt_tensor, src_mask, tgt_mask)
      output = rearrange(output, 's b d -> b d s')
      target = rearrange(target, 's b -> b s')

      loss = F.cross_entropy(output, target)
      losses.append(loss.item())
      pbar.set_description(f'Epoch {epoch_idx} | Loss: {loss.item():.3f}')
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      curr_idx += len(batch)
      if val_loader is not None and curr_idx - last_eval >= config['eval_freq']:
        last_eval = curr_idx
        eval_losses = eval(model, val_loader, preprocess_text, config)
        print(f'Epoch {epoch_idx} | Sample #{curr_idx} | Eval loss: {np.mean(eval_losses):.3f}')

        if 'model_save_path' in config:
          torch.save(model.state_dict(), config['model_save_path'])

  return losses


if __name__ == '__main__':
  # Load args and config
  args = parser.parse_args()
  config = load_config()
  
  # Get data loaders
  train_loader, val_loader, test_loader = get_dataloaders(
    config['beatmap_path'], batch_size=config.get('batch_size'))
  preprocess_text, vocab = get_text_preprocessor(config)

  # Create the model and load when applicable
  model = DefaultTransformer(
    n_token = len(vocab),
    d_model = config['d_model'],
    n_head = config['n_head'],
    d_hid = config['d_hid'],
    n_encoder_layers = config['n_encoder_layers'],
    n_decoder_layers = config['n_decoder_layers'],
    dropout = config['dropout']
  ).to(config['device'])
  
  if config['load_model'] and os.path.exists(config['model_save_path']):
    model.load_state_dict(torch.load(config['model_save_path']))

  optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

  # Train the model
  losses = train(model, train_loader, optimizer, preprocess_text, config, val_loader=val_loader)

  # Save the final model
  if 'model_save_path' in config:
    torch.save(model.state_dict(), config['model_save_path'])
  print('Model saved!')
    
  