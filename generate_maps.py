import argparse
import os
import shutil
import re

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm

from models import gen_seq_mask, model_from_config
from preprocessing.data_loading import *
from preprocessing.text_processing import get_text_preprocessor, prepare_tensor_seq
from preprocessing.text_processing import prepare_tensor_seqs
from utils import load_config, parse_args


parser = argparse.ArgumentParser()


AUDIO_FILE_NAME = 'audio.mp3'
MAX_MAP_LENGTH = 10000


def get_metadata_value(map_data, key):
  # Match key with no case
  match = re.findall(rf'{key}:(.*)$', map_data, re.IGNORECASE | re.MULTILINE)
  return match[0].strip() if match else None

def replace_metadata_value(map_data, key, value):
  return re.sub(rf'{key}:(.*)$', f'{key}:{value}', map_data, flags=re.MULTILINE)


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    
    map_path = os.path.join(config['beatmap_path'], 'song_mapping.csv')
    song_mapping = pd.read_csv(map_path, index_col=0)
    song_mapping = song_mapping.to_dict()['song']

    if args.map_ids is not None:
      map_ids = args.map_ids
    elif args.n_maps is not None:
      map_ids = np.random.choice(list(song_mapping.keys()), args.n_maps)

    dataset = OsuDataset(config['beatmap_path'], map_ids=map_ids)
    preprocess_text, vocab = get_text_preprocessor(config)
    model = model_from_config(config, vocab)

    hit_object_token = preprocess_text('<HitObject>')[0]
    pad_token = preprocess_text('<pad>')[0]
    start_token = preprocess_text(HIT_OBJECT_START_TOKEN)[0]
    end_token = preprocess_text(HIT_OBJECT_END_TOKEN)[0]

    # For each beatmap
    #   Load beatmap data
    #   Preprocess and to the training but sequentially
    #   Convert the output to a beatmap
    #   Write the new beatmap string
    #   Save as file along with audio in output directory
    for beatmap_idx, beatmap in enumerate(dataset):
      metadata, time_points, hit_objects, audio_data = beatmap
      src, tgt, audio_segments = format_training_data(
        metadata, time_points, hit_objects, audio_data, config)
      tgt = HIT_OBJECT_START_TOKEN

      # Uncomment to use the first real hit object
      # tgt = tgt[:tgt.find('<HitObject>', 1)] + '<HitObject>'

      src_tensor, src_mask = prepare_tensor_seq(
        [src], config['max_src_len'], preprocess_text, config)
      tgt_tensor, tgt_mask = prepare_tensor_seq(
        [tgt], config['max_gen_len'], preprocess_text, config, pad=False)
      
      # Pass the data through the model
      full_tgt = tgt_tensor.squeeze().cpu().tolist()
      if not isinstance(full_tgt, list):
        full_tgt = [full_tgt]
      n_hit_notes = tgt.count('<HitObject>')
      with torch.no_grad():
        for _ in tqdm(range(MAX_MAP_LENGTH)):
          # Generate the next token
          output = model(src_tensor, tgt_tensor, src_mask, tgt_mask)[-1]
          probs = F.softmax(output, dim=-1).squeeze()
          next_token = torch.multinomial(probs, 1).unsqueeze(0)

          full_tgt.append(next_token.item())
          if next_token.item() in (pad_token, end_token):
            break # Stop generation if we hit the end token or padding

          tgt_tensor = torch.cat([tgt_tensor, next_token], dim=0)
          tgt_tensor = tgt_tensor[-config['max_gen_len']:]
          tgt_mask = gen_seq_mask(tgt_tensor.shape[0]).to(config['device'])
      
      hit_objects = []
      itos = vocab.get_itos()
      for token in full_tgt:
        string = itos[token]
        if string == HIT_OBJECT_START_TOKEN:
          continue
        elif string == '<HitObject>':
          hit_objects.append(string)
        elif string == '<pad>' or string == HIT_OBJECT_END_TOKEN:
          break
        else:
          hit_objects[-1] += string

      # Figure out if we need to cut off a partial last hit object
      if itos[full_tgt[-1]] != HIT_OBJECT_END_TOKEN:
        hit_objects = hit_objects[:-1]

      hit_object_str = '\n'.join(hit_objects)

      hit_object_str = hit_object_str.replace('<HitObject>', '')
      hit_object_str = hit_object_str.replace(HIT_OBJECT_START_TOKEN, '')
      hit_object_str = hit_object_str.replace(HIT_OBJECT_END_TOKEN, '')

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