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
from vqvae.dataset import get_vqvae_dataloaders, sample_hitobjects, modify_timediff
from preprocessing.data_loading import format_training_data
from preprocessing.text_processing import get_text_preprocessor, prepare_tensor_seqs
from vqvae.tools import prepare_tensor_vqvae
from utils import load_config, parse_args, log
from vqvae.vqvae_model import VQEncoder, VQDecoder, Quantizer
import codecs as cs

# Create arguments
parser = argparse.ArgumentParser()
# parser.add_argument('--batch_size', type=int, default=32)


BEATMAP_PATH = 'data/formatted_beatmaps/'


def build_model(input_size, enc_channels, dec_channels, config):
  vq_encoder = VQEncoder(input_size - 1, enc_channels, config["n_down"])
  vq_decoder = VQDecoder(config['dim_vq_latent'], dec_channels, config['n_resblk'], config["n_down"])
  quantizer = Quantizer(config['codebook_size'], config['dim_vq_latent'], config['lambda_beta'])

  if config['load_model'] and os.path.exists(config['model_vqvae_save_path']):
    checkpoint = torch.load(config['model_vqvae_save_path'])
    vq_encoder.load_state_dict(checkpoint['vq_encoder'])
    vq_decoder.load_state_dict(checkpoint['vq_decoder'])
    quantizer.load_state_dict(checkpoint['quantizer'])
  return vq_encoder, vq_decoder, quantizer

def eval(encoder, decoder, quantizer, data_loader, preprocess_text, config):
  losses = []
  for batch in tqdm(data_loader):
    hitobjects = [obj[1] for obj in batch]
    batch_samples = [sample_hitobjects(obj) for obj in hitobjects]
    # src = training_samples
    src_tensor = prepare_tensor_vqvae(batch_samples, preprocess_text, config)
    src_tensor = src_tensor.detach().to(config['device']).float()
    target = src_tensor
    with torch.no_grad():
      pre_latents = encoder(src_tensor[..., :-1])
      embedding_loss, vq_latents, _, perplexity = quantizer(pre_latents)
      output = decoder(vq_latents)
      rec_loss = F.l1_loss(output, target)
      # loss = F.cross_entropy(output, target)
      loss = rec_loss + embedding_loss
      losses.append(loss.item())

  return losses
  

def train(encoder, decoder, quantizer, train_loader, preprocess_text, config, val_loader=None):

  encoder.to(config['device'])
  quantizer.to(config['device'])
  decoder.to(config['device'])

  opt_vq_encoder = torch.optim.Adam(encoder.parameters(), lr=config['lr'])
  opt_quantizer = torch.optim.Adam(quantizer.parameters(), lr=config['lr'])
  opt_vq_decoder = torch.optim.Adam(decoder.parameters(), lr=config['lr'])

  opt_vq_encoder_lr = torch.optim.lr_scheduler.ExponentialLR(opt_vq_encoder, gamma=0.98)
  opt_quantizer_lr = torch.optim.lr_scheduler.ExponentialLR(opt_quantizer, gamma=0.98)
  opt_vq_decoder_lr = torch.optim.lr_scheduler.ExponentialLR(opt_vq_decoder, gamma=0.98)
        
  
  last_eval = 0
  curr_idx = 0

  losses = []
  for epoch_idx in range(config['epochs']):
    for batch in (pbar := tqdm(train_loader)):
      hitobjects = [obj[1] for obj in batch]
      batch_samples = [sample_hitobjects(obj) for obj in hitobjects]
      # training_samples  = [format_training_data(None, None, obj, None)[1] for obj in batch_samples]

      # src = training_samples
      src_tensor = prepare_tensor_vqvae(batch_samples, preprocess_text, config)

      encoder.train()
      decoder.train()
      quantizer.train()
      src_tensor = src_tensor.detach().to(config['device']).float()
      target = src_tensor
      pre_latents = encoder(src_tensor[..., :-1])
      embedding_loss, vq_latents, _, perplexity = quantizer(pre_latents)
      output = decoder(vq_latents)
      
    #   output = rearrange(output, 's b d -> b d s')
    #   target = rearrange(target, 's b -> b s')

      rec_loss = F.l1_loss(output, target)
      # loss = F.cross_entropy(output, target)
      loss = rec_loss + embedding_loss

      losses.append(loss.item())
      pbar.set_description(f'Epoch {epoch_idx} | Loss: {loss.item():.3f}')
      log({'epoch': epoch_idx, 'train_loss': losses[-1]}, config)

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
        log({'epoch': epoch_idx, 'eval_loss': np.mean(eval_losses)}, config)

        if 'model_vqvae_save_path' in config:
          save(encoder, decoder, quantizer, config)

    if epoch_idx % config['lr_scheduler_e'] == 0:
      opt_vq_encoder_lr.step()
      opt_quantizer_lr.step()
      opt_vq_decoder_lr.step()

  return losses

def tokenize(encoder, quantizer, dataloader, preprocess_text, config):
  for batch in (pbar := tqdm(dataloader)):
    hitobjects = [obj[1] for obj in batch]
    batch_samples = [modify_timediff(obj) for obj in hitobjects]
    # training_samples  = [format_training_data(None, None, obj, None)[1] for obj in batch_samples]

    # src = training_samples
    src_tensor = prepare_tensor_vqvae(batch_samples, preprocess_text, config)

    encoder.eval()
    decoder.eval()
    quantizer.eval()
    src_tensor = src_tensor.detach().to(config['device']).float()
    pre_latents = encoder(src_tensor[..., :-1])
    indices = quantizer.map2index(pre_latents)
    indices = list(indices.cpu().numpy())
    indices = [str(token) for token in indices]

    os.makedirs(pjoin(config['beatmap_path'], config.get('token_save_path')), exist_ok = True)
    for line in batch:
        map_id, _ = line
        filename = '%s.txt' % (map_id)
        with cs.open(pjoin(config['beatmap_path'], config.get('token_save_path'), filename), 'a+') as f:
            # how to store token files? should it be stored in the same path of the beatmap file?
            f.write(' '.join(indices))
            f.write('\n')
    

def save(encoder, decoder, quantizer, config):
  state = {
    'vq_encoder': encoder.state_dict(),
    'quantizer': quantizer.state_dict(),
    'vq_decoder': decoder.state_dict()
  }
  torch.save(state, config['model_vqvae_save_path'])

if __name__ == '__main__':
  # Load args and config
  args = parse_args()
  config = load_config(args.config)
  if config['use_wandb']:
    import wandb
    wandb.init(project=config['wandb_project'], config=config)
  # Get data loaders
  train_loader, val_loader, test_loader = get_vqvae_dataloaders(
    config['beatmap_path'], batch_size=config.get('batch_size'), val_split = config.get('val_split'), test_split = config.get('test_split'))
  preprocess_text, vocab = get_text_preprocessor(config)
  
  enc_channels = [256, config.get('dim_vq_latent')]
  dec_channels = [config.get('dim_vq_latent'), 256, config.get('input_size')]
  # Create the model and load when applicable
  encoder, decoder, quantizer = build_model(config.get('input_size'), enc_channels, dec_channels, config)

  # Train the model
  losses = train(encoder, decoder, quantizer, train_loader, preprocess_text, config, val_loader=val_loader)

  # Save the final model
  if 'model_vqvae_save_path' in config:
    save(encoder, decoder, quantizer, config)
  print('Model saved!')

  # TODO tokenize all hitobjects
  tokenize_dataset, _, _ = get_vqvae_dataloaders(config['beatmap_path'], batch_size=config.get('batch_size'), val_split=0, test_split=0)
  tokenize(encoder, quantizer, tokenize_dataset, preprocess_text, config)
  