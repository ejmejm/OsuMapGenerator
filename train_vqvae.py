import argparse
import os
from os.path import join as pjoin
from unittest import skip

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
from models.vqvae import VQEncoder, VQDecoder, Quantizer
# Create arguments
parser = argparse.ArgumentParser()
# parser.add_argument('--batch_size', type=int, default=32)


BEATMAP_PATH = 'data/formatted_beatmaps/'


def build_model(input_size, enc_channels, dec_channels, config):
  vq_encoder = VQEncoder(input_size - 4, enc_channels, config["n_down"])
  vq_decoder = VQDecoder(config['dim_vq_latent'], dec_channels, config['n_resblk'], config["n_down"])
  quantizer = Quantizer(config['codebook_size'], config['dim_vq_latent'], config['lambda_beta'])

  if config['load_model'] and os.path.exists(config['model_save_path']):
    checkpoint = torch.load(config['model_save_path'])
    vq_encoder.load_state_dict(checkpoint['vq_encoder'])
    vq_decoder.load_state_dict(checkpoint['vq_decoder'])
    quantizer.load_state_dict(checkpoint['quantizer'])
  return vq_encoder, vq_decoder, quantizer

def eval(encoder, decoder, quantizer, data_loader, preprocess_text, config):
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
      pre_latents = encoder(src_tensor[..., :-4])
      embedding_loss, vq_latents, _, perplexity = quantizer(pre_latents)
      output = decoder(vq_latents)
    output = rearrange(output, 's b d -> b d s')
    target = rearrange(target, 's b -> b s')

    with torch.no_grad():
      rec_loss = F.l1_loss(output, target)
      loss = F.cross_entropy(output, target)
      loss += loss + rec_loss + embedding_loss
    losses.append(loss.item())

  return losses
  

def train(encoder, decoder, quantizer, train_loader, optimizer, preprocess_text, config, val_loader=None):
  opt_vq_encoder = torch.optim.Adam(encoder.parameters(), lr=config['lr'])
  opt_quantizer = torch.optim.Adam(quantizer.parameters(), lr=config['lr'])
  opt_vq_decoder = torch.optim.Adam(decoder.parameters(), lr=config['lr'])
  
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

      encoder.train()
      decoder.train()
      quantizer.train()

      pre_latents = encoder(src_tensor[..., :-4])
      embedding_loss, vq_latents, _, perplexity = quantizer(pre_latents)
      output = decoder(vq_latents)
      
      output = rearrange(output, 's b d -> b d s')
      target = rearrange(target, 's b -> b s')

      rec_loss = F.l1_loss(output, target)
      loss = F.cross_entropy(output, target)
      loss += loss + rec_loss + embedding_loss

      losses.append(loss.item())
      pbar.set_description(f'Epoch {epoch_idx} | Loss: {loss.item():.3f}')
      
      opt_vq_encoder.zero_grad()
      opt_quantizer.zero_grad()
      opt_vq_decoder.zero_grad()

      loss.backward()

      opt_vq_encoder.step()
      opt_quantizer.step()
      opt_vq_decoder.step()

      curr_idx += len(batch)
      if val_loader is not None and curr_idx - last_eval >= config['eval_freq']:
        last_eval = curr_idx
        eval_losses = eval(encoder, decoder, quantizer, val_loader, preprocess_text, config)
        print(f'Epoch {epoch_idx} | Sample #{curr_idx} | Eval loss: {np.mean(eval_losses):.3f}')

        if 'model_save_path' in config:
          save(encoder, decoder, quantizer, config)

  return losses

def tokenize(encoder, quantizer, dataloader, config):
  for batch in (pbar := tqdm(dataloader)):
    pre_latents = encoder(batch[..., :-4])
    indices = quantizer.map2index(pre_latents)
    indices = list(indices.cpu().numpy())
    indices = [str(token) for token in indices]
    with os.open(pjoin(config['token_save_path'], '.txt'), 'a+') as f:
        # how to store token files? should it be stored in the same path of the beatmap file?
        skip
    

def save(encoder, decoder, quantizer, config):
  state = {
    'vq_encoder': encoder.state_dict(),
    'quantizer': quantizer.state_dict(),
    'vq_decoder': decoder.state_dict()
  }
  torch.save(state, config['model_save_path'])

if __name__ == '__main__':
  # Load args and config
  args = parser.parse_args()
  config = load_config()
  
  # Get data loaders
  train_loader, val_loader, test_loader = get_dataloaders(
    config['beatmap_path'], batch_size=config.get('batch_size'))
  preprocess_text, vocab = get_text_preprocessor(config)

  # Create the model and load when applicable
  encoder, decoder, quantizer = build_model()

  # Train the model
  losses = train(encoder, decoder, quantizer, train_loader, preprocess_text, config, val_loader=val_loader)

  # Save the final model
  if 'model_save_path' in config:
    save(encoder, decoder, quantizer, config)
  print('Model saved!')

  # TODO tokenize all hitobjects
    
  