import argparse
import os

from clearml import Task
from einops import rearrange
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from vqvae.transformer import model_from_config
from vqvae.dataset import get_dataloaders
from preprocessing.data_loading import format_training_data, sample_from_map
from vqvae.dataset import sample_tokensets_from_map, format_metadata, sample_audio_and_tokens_from_map
from preprocessing.text_processing import get_text_preprocessor, prepare_tensor_seqs
from utils import load_config, log, parse_args
from vqvae.tools import prepare_tensor_transformer, prepare_tensor_vqvae, prepare_tensor_tokens
from vqvae.vqvae_model import VQEncoder, VQDecoder, Quantizer, build_model

# Create arguments
parser = argparse.ArgumentParser()
# parser.add_argument('--batch_size', type=int, default=32)


BEATMAP_PATH = 'data/formatted_beatmaps/'
MAX_HIT_OBJECTS = 100
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def eval(model, data_loader, preprocess_text, config):
  losses = []
  model.eval()
  for batch in tqdm(data_loader):
    if not config.get('use_vqvae'):
      batch_samples = [sample_from_map(*map) for map in batch]
      training_samples = [format_training_data(*map, config) for map in batch_samples]

      src, tgt = zip(*training_samples)
      # Convert text to numerical tensors with padding and corresponding masks
      src_tensor, tgt_tensor, src_mask, tgt_mask = \
        prepare_tensor_seqs(src, tgt, preprocess_text, config)
      # Split the tgt tensor into the input and actual target
      target = tgt_tensor[1:]
      tgt_tensor = tgt_tensor[:-1]
      tgt_mask = tgt_mask[:-1, :-1]

      # Pass the data through the model
      output = model(src_tensor, tgt_tensor, src_mask, tgt_mask)
      # Rearrange data to be batch first
      output = rearrange(output, 's b d -> b d s')
      target = rearrange(target, 's b -> b s')
    else:
      # why not processing these things in the dataset?
      batch_samples = [sample_tokensets_from_map(config, *map) for map in batch]
      training_samples = [format_metadata(*map) for map in batch_samples]

      meta, tokens, audio = zip(*training_samples)
      # Convert text to numerical tensors with padding and corresponding masks
      src_tensor, tgt_tensor, src_mask, tgt_mask = \
        prepare_tensor_transformer(meta, audio, tokens, preprocess_text, config)
      # Split the tgt tensor into the input and actual target
      target = tgt_tensor[:, 1:]
      gt_tensor = tgt_tensor[:, :-1]

      output = model(src_tensor, gt_tensor, audio = audio)

      output = output.view(-1, output.shape[-1]).clone()
      target = target.contiguous().view(-1).clone()

    with torch.no_grad():
      loss = F.cross_entropy(output, target)
    losses.append(loss.item())

  return losses

def cal_performance(pred, gold, trg_pad_idx):
  loss = cal_loss(pred, gold, trg_pad_idx)
  pred = pred.max(1)[1]
  gold = gold.contiguous().view(-1)
  non_pad_mask = gold.ne(trg_pad_idx)
  n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
  n_word = non_pad_mask.sum().item()
  return loss, pred, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx):
  '''Calculate cross entropy loss, apply label smoothing if needed.'''
  loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
  return loss

  
def train(model, encoder, quantizer, train_loader, optimizer, preprocess_text, config, val_loader=None):
  last_eval = 0
  curr_idx = 0
  model.to(config['device'])
  encoder.to(config['device'])
  quantizer.to(config['device'])
  losses = []
  opt_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
  for epoch_idx in range(config['epochs']):
    for batch in (pbar := tqdm(train_loader)):
      model.train()
      if not config.get('use_vqvae'):
        batch_samples = [sample_from_map(*map) for map in batch]
        training_samples = [format_training_data(*map, config) for map in batch_samples]

        src, tgt = zip(*training_samples)
        # Convert text to numerical tensors with padding and corresponding masks
        src_tensor, tgt_tensor, src_mask, tgt_mask = \
          prepare_tensor_seqs(src, tgt, preprocess_text, config)
        # Split the tgt tensor into the input and actual target
        target = tgt_tensor[1:]
        tgt_tensor = tgt_tensor[:-1]
        tgt_mask = tgt_mask[:-1, :-1]

        # Pass the data through the model
        output = model(src_tensor, tgt_tensor, src_mask, tgt_mask)
        # Rearrange data to be batch first
        output = rearrange(output, 's b d -> b d s')
        target = rearrange(target, 's b -> b s')
      else:
        # why not processing these things in the dataset?
        batch_samples = [sample_audio_and_tokens_from_map(config, *map, audio_secs = 10) for map in batch]
        training_samples = [format_metadata(*map) for map in batch_samples]

        meta, hit_objects, audio = zip(*training_samples)
        audio = torch.from_numpy(np.stack(audio)).float()
        audio = audio.to(config['device'])
        audio = audio.transpose(1, 2)
        token_tensor = prepare_tensor_tokens(hit_objects, encoder, quantizer, config['input_size'], preprocess_text, config)
        
        # Convert text to numerical tensors with padding and corresponding masks
        src_tensor, tgt_tensor, src_mask, tgt_mask = \
          prepare_tensor_transformer(meta, audio, token_tensor, preprocess_text, config)
        # Split the tgt tensor into the input and actual target
        target = tgt_tensor[:, 1:]
        gt_tensor = tgt_tensor[:, :-1]

        output = model(src_tensor, gt_tensor, audio = audio)
      # Calculate loss
      o = output.view(-1, output.shape[-1]).clone()
      gt = target.contiguous().view(-1).clone()
      loss, pred_seq, n_correct, n_word = cal_performance(o, gt, config['codebook_size'] + 2)
      losses.append(loss.item())
      print(target[0])
      print(pred_seq.view(target.shape)[0])
      pbar.set_description(f'Epoch {epoch_idx} | Loss: {loss.item()/n_word:.3f}')
      log({'epoch': epoch_idx, 'train_loss': losses[-1]}, config)
      
      # Backprop
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      curr_idx += len(batch)
      # Eval
      if val_loader is not None and curr_idx - last_eval >= config['eval_freq']:
        last_eval = curr_idx
        eval_losses = eval(model, val_loader, preprocess_text, config)
        print(f'Epoch {epoch_idx} | Sample #{curr_idx} | Eval loss: {np.mean(eval_losses):.3f}')
        log({'epoch': epoch_idx, 'eval_loss': np.mean(eval_losses)}, config)

        if 'model_save_path' in config:
          torch.save(model.state_dict(), config['model_save_path'])
      if curr_idx % config['lr_scheduler_e'] == 0:
        opt_lr.step()
  return losses


if __name__ == '__main__':
  # Load args and config
  args = parse_args()
  config = load_config(args.config)

  if config['use_wandb']:
    import wandb
    wandb.init(project=config['wandb_project'], config=config)
  
  # Get data loaders
  train_loader, val_loader, test_loader = get_dataloaders(config, 
    config['beatmap_path'], batch_size=config.get('batch_size'))
  preprocess_text, vocab = get_text_preprocessor(config)

  enc_channels = [config.get('dim_vq_latent')]
  dec_channels = [config.get('dim_vq_latent'), config.get('input_size')]
  encoder, decoder, quantizer = build_model(config.get('input_size'), enc_channels, dec_channels, config)
  
  # Create model and load when applicable
  model = model_from_config(config, vocab)
  print('# params:', sum(p.numel() for p in model.parameters()))
  optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

  # Train the model
  try:
    losses = train(model, encoder, quantizer, train_loader, optimizer, preprocess_text, config, val_loader=val_loader)
  except KeyboardInterrupt:
    print('Training interrupted.')

  # Save the final model
  if 'model_save_path' in config:
    torch.save(model.state_dict(), config['model_save_path'])
  print('Model saved!')