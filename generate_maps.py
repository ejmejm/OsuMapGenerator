import argparse
import os
import shutil
import re
import time

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
MAX_MAP_LENGTH = 800


def get_metadata_value(map_data, key):
  # Match key with no case
  match = re.findall(rf'{key}:(.*)$', map_data, re.IGNORECASE | re.MULTILINE)
  return match[0].strip() if match else None

def replace_metadata_value(map_data, key, value):
  return re.sub(rf'{key}:(.*)$', f'{key}:{value}', map_data, flags=re.MULTILINE)


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)

    torch.manual_seed(time.time())
    
    map_path = os.path.join(config['beatmap_path'], 'song_mapping.csv')
    song_mapping = pd.read_csv(map_path, index_col=0)
    song_mapping = song_mapping.to_dict()['song']

    if args.map_ids is not None:
      map_ids = args.map_ids
    elif args.n_maps is not None:
      map_ids = np.random.choice(list(song_mapping.keys()), args.n_maps)

    dataset = OsuDataset(config, map_ids=map_ids)
    preprocess_text, vocab = get_text_preprocessor(config)
    model = model_from_config(config, vocab)
    itos = vocab.get_itos()

    hit_object_token = preprocess_text('<HitObject>')[0]
    pad_token = preprocess_text('<pad>')[0]
    start_token = preprocess_text(HIT_OBJECT_START_TOKEN)[0]
    end_token = preprocess_text(HIT_OBJECT_END_TOKEN)[0]
    break_token = preprocess_text(BREAK_TOKEN)[0]
    audio_token = preprocess_text(AUDIO_SEGMENT_TOKEN)[0]
    audio_placeholder_token = preprocess_text(AUDIO_PLACEHOLDER_TOKEN)[0]
    ho_start_tokens = set([start_token, hit_object_token, break_token])

    # For each beatmap
    #   Load beatmap data
    #   Preprocess and to the training but sequentially
    #   Convert the output to a beatmap
    #   Write the new beatmap string
    #   Save as file along with audio in output directory
    for beatmap_idx, beatmap in enumerate(dataset):
      metadata, time_points, hit_objects, audio_data = beatmap
      f_time_points = format_time_points(time_points)[0]

      src, tgt, audio_segments = format_training_data(
        metadata, time_points, hit_objects, audio_data, config)

      use_first_hit_object = False
      if use_first_hit_object:
        tgt = tgt[:tgt.find('<HitObject>', 1)] + '<HitObject>'
        n_start_segments = tgt.count(AUDIO_SEGMENT_TOKEN)
        audio_segments = audio_segments[:n_start_segments] # (placeholder_id, seq, channels)
        # TODO: This assumes the this all happens during the first time point
        # That may not always be the case, and will be wrong those times
        beat_len = tp_to_beat_len(f_time_points[0])
        curr_time = tgt.count(BREAK_TOKEN) * beat_len
      else:
        tgt = HIT_OBJECT_START_TOKEN
        audio_segments = audio_segments[:0]
        curr_time = 0

      src_tensor, src_mask = prepare_tensor_seq(
        [src], config['max_src_len'], preprocess_text, config)
      tgt_tensor, tgt_mask = prepare_tensor_seq(
        [tgt], config['max_gen_len'], preprocess_text, config, pad=False)


      # Pass the data through the model
      full_tgt = tgt_tensor.squeeze().cpu().tolist()
      if not isinstance(full_tgt, list):
        full_tgt = [full_tgt]
      tp_index = 0 # Keep track of the current time point
      curr_time = 0 # In milliseconds


      with torch.no_grad():
        # Start looping through predicted tokens
        for _ in tqdm(range(MAX_MAP_LENGTH)):
          # print('---------------', curr_time, '---------------')
          # Generate the next token
          if config['include_audio']:
            f_audio_segments = prepare_audio_tensor([audio_segments], config)
            audio_idxs = tgt_tensor == audio_placeholder_token
            output = model(
              src_tensor, tgt_tensor, src_mask, tgt_mask,
              audio=f_audio_segments, audio_mask=audio_idxs)[-1]
          else:
            output = model(src_tensor, tgt_tensor, src_mask, tgt_mask)[-1]

          # Zero the probability for audio placeholder tokens
          output[:, audio_placeholder_token] = -float('inf')
          probs = F.softmax(output, dim=-1).squeeze()
          next_token = torch.multinomial(probs, 1).unsqueeze(0)

          full_tgt.append(next_token.item())
          # print(''.join([itos[x] for x in full_tgt]))
          if next_token.item() in (pad_token, end_token):
            break # Stop generation if we hit the end token or padding


          ### Audio Stuff ###


          elif next_token.item() == audio_token and config['include_audio']:
            # When this happens, we need to grab audio for the segment
            # And then generate new placeholder tokens for it

            # Now get audio segment data

            # Find the previous hit object or break or start token
            # Using that info, get the number of beats since the last hit object
            #   
            # If hit object:
            #   If the next time point exists:
            #     Use that beat_len to calculate a possible time for the hit object
            #     If the time is after the next time point:
            #       Use that time
            #       Increment the time point index
            #
            # If no hit object time has been decided yet:
            #   Use the current time point to get beat_len
            #   Use that beat_len to calculate a time for the hit object

            # Get most recent hit object (including breaks and start)
            for i in range(len(full_tgt)-1, -1, -1):
              if full_tgt[i] in ho_start_tokens:
                break
            
            if not config['relative_timing']:
              raise NotImplementedError('Generation currently only supports relative time')

            # Get number of beats since last hit
            if full_tgt[i] == start_token:
              beats_since_last = 0
            elif full_tgt[i] == break_token:
              beats_since_last = config['break_length']
            elif full_tgt[i] == hit_object_token:
              hit_object_str = ''.join([itos[x] for x in full_tgt[i+1:-1]])
              beat_str = hit_object_str.split(',')[2]
              beats_since_last = get_relative_time_beats(beat_str)
            else:
              raise ValueError(f'Invalid token found in full_tgt, {full_tgt[i]}')

            # Check if we need to switch to the next timepoint
            found_time = False
            if full_tgt[i] == hit_object_token and tp_index + 1 < len(f_time_points):
              next_beat_len = tp_to_beat_len(f_time_points[tp_index+1])
              next_time = curr_time + beats_since_last * next_beat_len
              next_tp_time = tp_to_time(f_time_points[tp_index+1])
              if next_time >= next_tp_time:
                curr_time = next_time
                beat_len = next_beat_len
                found_time = True
                tp_index += 1
                  
            # Calcualte the time with the current timepoint
            # if we haven't found a time yet
            if not found_time:
              beat_len = tp_to_beat_len(f_time_points[tp_index])
              curr_time += beats_since_last * beat_len

            # Get the audio segment

            start_idx = int(curr_time * config['sample_rate'] / 1000)
            end_idx = int((curr_time + beat_len * config['break_length']) * config['sample_rate'] / 1000)
            segment = audio_data[start_idx:end_idx]

            if end_idx > len(audio_data):
              segment = np.concatenate([segment, np.zeros((end_idx - len(audio_data),))])
            mel_bands = audio_to_np(
              segment, config['segments_per_beat'] * config['break_length'],
              n_mel_bands=config['n_mel_bands'], sample_rate=config['sample_rate'])

            if audio_segments is None:
              audio_segments = mel_bands.unsqueeze(0) # (placeholder_id, seq, channels)
            else:
              audio_segments = np.concatenate(
                [audio_segments, mel_bands[None]], axis=0)


            # Finally, add the audio palceholder tokens to the target
            audio_token_set = [audio_token]
            for _ in range(config['audio_tokens_per_segment']):
              full_tgt.append(audio_placeholder_token)
              audio_token_set.append(audio_placeholder_token)
            audio_token_set = torch.tensor(audio_token_set).unsqueeze(1)
            next_token = audio_token_set.to(tgt_tensor.device)


          tgt_tensor = torch.cat([tgt_tensor, next_token], dim=0)

          n_clip = len(tgt_tensor) - config['max_tgt_len']
          orig_clip = n_clip
          if n_clip > 0:
            for i in range(config['audio_tokens_per_segment']):
              if tgt_tensor[orig_clip+i].item() == audio_placeholder_token:
                n_clip += 1
              elif tgt_tensor[orig_clip+i].item() == audio_token:
                break

            clipped = tgt_tensor[:n_clip]
            audio_n_clip = (clipped == audio_token).sum().item()
            audio_segments = audio_segments[audio_n_clip:]

            tgt_tensor = tgt_tensor[n_clip:]


          tgt_mask = gen_seq_mask(tgt_tensor.shape[0]).to(config['device'])
      

      ### Generation Complete ###


      hit_objects = []
      for token in full_tgt:
        string = itos[token]
        if string == '<HitObject>':
          hit_objects.append('')
        elif string == BREAK_TOKEN:
          hit_objects.append(string)
        elif string == '<pad>' or string == HIT_OBJECT_END_TOKEN:
          break
        elif string.startswith('<'):
          continue
        else:
          hit_objects[-1] += string

      # Figure out if we need to cut off a partial last hit object
      if itos[full_tgt[-1]] != HIT_OBJECT_END_TOKEN:
        hit_objects = hit_objects[:-1]

      # Convert relative timing to absolute timing
      if config['relative_timing']:
        tp_index = 0 # Keep track of the current time point
        curr_time = 0 # In milliseconds

        for i in range(len(hit_objects)):
          # Get number of beats since last hit
          if hit_objects[i] == BREAK_TOKEN:
            beat_len = tp_to_beat_len(f_time_points[tp_index])
            curr_time += config['break_length'] * beat_len
          else:
            ho_parts = hit_objects[i].split(',')
            beat_str = ho_parts[2]
            beats_since_last = get_relative_time_beats(beat_str)

            # Check if we need to switch to the next timepoint
            found_time = False
            if tp_index + 1 < len(f_time_points):
              next_beat_len = tp_to_beat_len(f_time_points[tp_index+1])
              next_time = curr_time + beats_since_last * next_beat_len
              next_tp_time = tp_to_time(f_time_points[tp_index+1])
              if next_time >= next_tp_time:
                curr_time = next_time
                beat_len = next_beat_len
                found_time = True
                tp_index += 1
                  
            # Calcualte the time with the current timepoint
            # if we haven't found a time yet
            if not found_time:
              beat_len = tp_to_beat_len(f_time_points[tp_index])
              curr_time += beats_since_last * beat_len

            hit_objects[i] = ','.join(ho_parts[:2] + [str(int(curr_time))] + ho_parts[3:])

      hit_object_str = '\n'.join(hit_objects)

      hit_object_str = hit_object_str.replace('<HitObject>', '')
      hit_object_str = hit_object_str.replace(BREAK_TOKEN, '')

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