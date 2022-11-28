from einops import rearrange
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from models import model_from_config
from preprocessing.data_loading import get_dataloaders
from preprocessing.text_processing import get_text_preprocessor
from utils import load_config, log, parse_args


BEATMAP_PATH = 'data/formatted_beatmaps/'
MAX_HIT_OBJECTS = 100


def eval(model, data_loader, preprocess_text, config):
  loss_hist = []
  model.eval()
  for batch in tqdm(data_loader):
    src_tensor, tgt_tensor, src_mask, tgt_mask, \
    audio_segments, audio_idxs, target = \
      [None if x is None else x.to(config['device']) for x in batch]

    with torch.no_grad():
      output = model(
        src_tensor, tgt_tensor, src_mask, tgt_mask,
        audio=audio_segments, audio_mask=audio_idxs)
    output = rearrange(output, 's b d -> b d s')
    target = rearrange(target, 's b -> b s')

    with torch.no_grad():
      losses = F.cross_entropy(output, target, reduction='none')

    if config['include_audio']:
      audio_idxs = rearrange(audio_idxs, 's b -> b s')
      audio_mask = ~audio_idxs
      losses = losses.masked_select(audio_mask)

    loss = losses.mean()
    loss_hist.append(loss.item())

  return loss_hist
  

def train(model, train_loader, optimizer, preprocess_text, config, val_loader=None):
  last_eval = 0
  curr_idx = 0

  loss_hist = []
  for epoch_idx in range(config['epochs']):
    for batch in (pbar := tqdm(train_loader)):
      model.train()
      src_tensor, tgt_tensor, src_mask, tgt_mask, \
      audio_segments, audio_idxs, target = \
        [None if x is None else x.to(config['device']) for x in batch]

      # Pass the data through the model
      output = model(
        src_tensor, tgt_tensor, src_mask, tgt_mask,
        audio=audio_segments, audio_mask=audio_idxs)
        
      # Rearrange data to be batch first
      output = rearrange(output, 's b d -> b d s')
      target = rearrange(target, 's b -> b s')

      # Calculate loss
      losses = F.cross_entropy(output, target, reduction='none')

      # Mask out losses for audio tokens
      if config['include_audio']:
        audio_idxs = rearrange(audio_idxs, 's b -> b s')
        audio_mask = ~audio_idxs
        losses = losses.masked_select(audio_mask)

      loss = losses.mean()
      
      loss_hist.append(loss.item())
      pbar.set_description(f'Epoch {epoch_idx} | Loss: {loss.item():.3f}')
      log({'epoch': epoch_idx, 'train_loss': loss_hist[-1]}, config)
      
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

  return losses


if __name__ == '__main__':
  # Load args and config
  args = parse_args()
  config = load_config(args.config)

  if config['use_wandb']:
    import wandb
    wandb.init(project=config['wandb_project'], config=config)

  if config['n_load_workers'] > 0:
    torch.multiprocessing.set_start_method('spawn', force=True)
  
  # Get data loaders
  preprocess_text, vocab = get_text_preprocessor(config)
  train_loader, val_loader, test_loader = get_dataloaders(
    config, preprocess=True)
  # print(vocab.get_itos())

  # Create model and load when applicable
  model = model_from_config(config, vocab)
  print('# params:', sum(p.numel() for p in model.parameters()))
  optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

  # Train the model
  try:
    losses = train(model, train_loader, optimizer, preprocess_text, config, val_loader=val_loader)
  except KeyboardInterrupt:
    print('Training interrupted.')

  # Save the final model
  if 'model_save_path' in config:
    torch.save(model.state_dict(), config['model_save_path'])
  print('Model saved!')

  # Test the model
  if test_loader is not None:
    test_losses = eval(model, test_loader, preprocess_text, config)
    print(f'Test loss: {np.mean(test_losses):.3f}')