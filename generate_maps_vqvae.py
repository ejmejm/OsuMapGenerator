import argparse
import os
import re
import shutil
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from vqvae.transformer import model_from_config
from vqvae.dataset import get_dataloaders, OsuTokensDataset
from vqvae.dataset import format_metadata
from preprocessing.text_processing import get_text_preprocessor
from utils import load_config, parse_args
from vqvae.tools import prepare_tensor_transformer, reconstruct_hitobjects
from vqvae.vqvae_model import VQEncoder, VQDecoder, Quantizer, build_model

# Create arguments
parser = argparse.ArgumentParser()
# parser.add_argument('--batch_size', type=int, default=32)

AUDIO_FILE_NAME = 'audio.mp3'
BEATMAP_PATH = 'data/formatted_beatmaps/'
MAX_HIT_OBJECTS = 100

def get_metadata_value(map_data, key):
  # Match key with no case
  match = re.findall(rf'{key}:(.*)$', map_data, re.IGNORECASE | re.MULTILINE)
  return match[0].strip() if match else None

def replace_metadata_value(map_data, key, value):
  return re.sub(rf'{key}:(.*)$', f'{key}:{value}', map_data, flags=re.MULTILINE)


if __name__ == '__main__':
  # Load args and config
  args = parse_args()
  config = load_config(args.config)
  
  # # Get data loaders
  # train_loader, val_loader, test_loader = get_dataloaders(config, 
  #   config['beatmap_path'], batch_size=1)
  preprocess_text, vocab = get_text_preprocessor(config)

  map_path = os.path.join(config['beatmap_path'], 'song_mapping.csv')
  song_mapping = pd.read_csv(map_path, index_col=0)
  song_mapping = song_mapping.to_dict()['song']

  if args.map_ids is not None:
    map_ids = args.map_ids
  elif args.n_maps is not None:
    map_ids = np.random.choice(list(song_mapping.keys()), args.n_maps)

  dataset = OsuTokensDataset(config['beatmap_path'], map_ids=map_ids)
  # Create model and load when applicable
  transformer = model_from_config(config, vocab)
  print('# params:', sum(p.numel() for p in transformer.parameters()))

  enc_channels = [config.get('dim_vq_latent')]
  dec_channels = [config.get('dim_vq_latent'), config.get('input_size')]
  encoder, decoder, quantizer = build_model(config.get('input_size'), enc_channels, dec_channels, config)

  decoder.to(config['device'])
  quantizer.to(config['device'])
  transformer.to(config['device'])
  for beatmap_idx, beatmap in enumerate(dataset):
    training_samples = [format_metadata(*map) for map in [beatmap]]

    meta, tokens, audio = zip(*training_samples)
    # Convert text to numerical tensors with padding and corresponding masks
    src_tensor, tgt_tensor, src_mask, tgt_mask = \
      prepare_tensor_transformer(meta, audio, tokens, preprocess_text, config)
    
    pred_tokens = transformer.sample(src_tensor, trg_sos=config['codebook_size'],
                                                     trg_eos=config['codebook_size'] + 1, max_steps=200, sample=False,
                                                     top_k=100)
    pred_tokens = pred_tokens[:, 1:]
    print(pred_tokens[0])
    if len(pred_tokens[0]) == 0:
        continue
    vq_latent = quantizer.get_codebook_entry(pred_tokens)
    hit_objects = decoder(vq_latent)
    hit_objects = hit_objects.detach().cpu().numpy()

    hit_objects = reconstruct_hitobjects(hit_objects)
    # break
    hit_object_str = '\n'.join([','.join(str(c) for c in line) for line in hit_objects[0]])  

    map_path = dataset.get_map_path(beatmap_idx)
    with open(map_path, 'r', encoding='utf-8') as f:
      beatmap_str = f.read()
    beatmap_str = beatmap_str[:beatmap_str.find('[HitObjects]')]
    beatmap_str += '[HitObjects]\n' + hit_object_str

    # Replace metadata
    beatmap_str = replace_metadata_value(beatmap_str, 'Version', 'AI')
    beatmap_str = replace_metadata_value(
      beatmap_str, 'AudioFilename', f' {AUDIO_FILE_NAME}')

    # Get metadata values needed for file naming
    version = get_metadata_value(beatmap_str, 'Version') or 'No Version'
    title = get_metadata_value(beatmap_str, 'Title') or 'No Title'
    author = get_metadata_value(beatmap_str, 'Artist') or 'No Author'
    creator = get_metadata_value(beatmap_str, 'Creator')
    rand_id = np.random.randint(1_000_000)

    # folder_name = ID author - title
    # version = AI
    # file name = author - title (creator) [version].osu
    # audio = audio.mp3

    output_folder_name = f'{rand_id} - {title}'
    output_path = os.path.join(config['output_dir'], output_folder_name)

    if creator:
      map_file_name = f'{author} - {title} ({creator}) [{version}].osu'
    else:
      map_file_name = f'{author} - {title} [{version}].osu'
    
    # Delete directory if it exists and remake it
    if os.path.exists(output_path):
      shutil.rmtree(output_path)
    os.makedirs(output_path)

    with open(os.path.join(output_path, map_file_name), 'w+', encoding='utf-8') as f:
      f.write(beatmap_str)

    # Copy audio to new directory
    src_map_file_name = dataset.map_list[beatmap_idx]
    src_audio_file_name = dataset.mapping[src_map_file_name]
    audio_src_path = os.path.join(config['beatmap_path'], 'songs', src_audio_file_name)
    shutil.copy(audio_src_path, os.path.join(output_path, AUDIO_FILE_NAME))