from decimal import Decimal, getcontext
import os
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from utils import load_config
from preprocessing.data_loading import load_beatmap_data, format_time_points, get_time_points_in_range
import random

DEFAULT_METADATA = set([
  'DistanceSpacing', # 'AudioLeadIn', 'Countdown', 'CountdownOffset', 
  'BeatDivisor', 'GridSize', 'CircleSize', 'OverallDifficulty', 'ApproachRate',
  'SliderMultiplier', 'SliderTickRate', 'HPDrainRate'
])

# TODO these datasets should be moved to a single python file.
class OsuVQVaeDataset(Dataset):
  def __init__(self, root_dir, include_audio=True):
    self.root_dir = root_dir
    self.include_audio = include_audio
    self.map_dir = os.path.join(root_dir, 'maps')
    self.audio_dir = os.path.join(root_dir, 'songs')
  
    # Mapping of map_id to song_id
    self.mapping = pd.read_csv(
      os.path.join(root_dir, 'song_mapping.csv'), index_col=0)
    self.mapping = self.mapping.to_dict()['song']
    self.map_list = list(self.mapping.keys())

  def __len__(self):
    # This returns the number of maps, not the actual number of samples
    return len(self.map_list)

  def __getitem__(self, idx):
    # Need to return selected metadata text, hitobject text, and audio data separately
    map_id = self.map_list[idx]
    _, _, hit_objects = load_beatmap_data(
      os.path.join(self.map_dir, map_id))

    return map_id, hit_objects

class OsuTokensDataset(Dataset):
  def __init__(self, root_dir, include_audio=True):
    self.root_dir = root_dir
    self.include_audio = include_audio
    self.map_dir = os.path.join(root_dir, 'maps')
    self.audio_dir = os.path.join(root_dir, 'songs')
    self.token_dir = os.path.join(root_dir, 'gen/tokens')
    # Mapping of map_id to song_id
    self.mapping = pd.read_csv(
      os.path.join(root_dir, 'song_mapping.csv'), index_col=0)
    self.mapping = self.mapping.to_dict()['song']
    self.map_list = list(self.mapping.keys())

  def __len__(self):
    # This returns the number of maps, not the actual number of samples
    return len(self.map_list)

  def __getitem__(self, idx):
    # Need to return selected metadata text, hitobject text, and audio data separately
    map_id = self.map_list[idx]
    metadata, time_points, hit_objects = load_beatmap_data(
      os.path.join(self.map_dir, map_id))
    if self.include_audio:
      # TODO: Add audio loading here
      audio_data = []
    else:
      audio_data = []

    tokens_list = load_token_data(os.path.join(self.token_dir, "%s.txt"%(map_id)))
    tokens = random.choice(tokens_list)
    tokens = [int(token) for token in tokens]
    return metadata, time_points, tokens, audio_data

# TODO: Add breaks to dataset
class OsuDataset(Dataset):
  def __init__(self, root_dir, include_audio=True, map_ids=None):
    self.root_dir = root_dir
    self.include_audio = include_audio
    self.map_dir = os.path.join(root_dir, 'maps')
    self.audio_dir = os.path.join(root_dir, 'songs')
  
    # Mapping of map_id to song_id
    self.mapping = pd.read_csv(
      os.path.join(root_dir, 'song_mapping.csv'), index_col=0)
    self.mapping = self.mapping.to_dict()['song']
    if map_ids is not None:
      new_mapping = {}
      for map_id in map_ids:
        new_mapping[map_id] = self.mapping[map_id]
      self.mapping = new_mapping
    self.map_list = list(self.mapping.keys())

  def __len__(self):
    # This returns the number of maps, not the actual number of samples
    return len(self.map_list)

  def get_map_path(self, idx):
    map_id = self.map_list[idx]
    return os.path.join(self.map_dir, map_id)

  def __getitem__(self, idx):
    # Need to return selected metadata text, hitobject text, and audio data separately
    map_id = self.map_list[idx]
    metadata, time_points, hit_objects = load_beatmap_data(
      os.path.join(self.map_dir, map_id))
    if self.include_audio:
      # TODO: Add audio loading here
      audio_data = []
    else:
      audio_data = []

    return metadata, time_points, hit_objects, audio_data

# Get dataloaders for training test and validation
def get_dataloaders(config, root_dir, batch_size=1, include_audio=True,
                    val_split=0.05, test_split=0.1, shuffle=True):
  if (config.get('use_vqvae')):
    dataset = OsuTokensDataset(root_dir, include_audio=include_audio)
  else:
    dataset = OsuDataset(root_dir, include_audio=include_audio)
  # Split dataset into train, val, and test
  val_size = int(val_split * len(dataset))
  test_size = int(test_split * len(dataset))
  train_size = len(dataset) - val_size - test_size

  train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size])

  collate_fn = lambda x: x
  train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
  if val_size != 0:
    val_loader = DataLoader(
      val_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
  else:
    val_loader = None
  if test_size != 0:
    test_loader = DataLoader(
      test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
  else:
    test_loader = None

  return train_loader, val_loader, test_loader

def sample_tokensets_from_map(
    metadata, time_points, tokens, audio_data,
    n_tokens=10, audio_secs=5, target_metadata=DEFAULT_METADATA):

  selected_metadata = {}
  for key in target_metadata:
    value = metadata.get(key)
    if value is not None:
      selected_metadata[key] = value

  start_idx = np.random.randint(0, max(1, len(tokens) - n_tokens))
  selected_tokens = tokens[start_idx:start_idx + n_tokens]
  selected_time_points = time_points

  # TODO: Add a way to sample a portion of audio data
  selected_audio = []

  return selected_metadata, selected_time_points, \
         selected_tokens, selected_audio

def format_metadata(metadata, time_points, tokens, audio_data):
  prior_str = '<Metadata>'
  for key, value in metadata.items():
    prior_str += f'<{key}>{value}'

  f_time_points, slider_changes = format_time_points(time_points)

  prior_str += '<TimePoints>'
  for tp in f_time_points:
    prior_str += f'<TimePointSplit>{tp}'

  if len(slider_changes) > 0:
    prior_str += '<SliderChanges>'
    for sc in slider_changes:
      prior_str += f'<SliderChange>{sc}'
  return prior_str, tokens, audio_data

def get_vqvae_dataloaders(root_dir, batch_size=1, include_audio=True,
                    val_split=0.05, test_split=0.1, shuffle=True):
  dataset = OsuVQVaeDataset(root_dir, include_audio=include_audio)
  val_size = int(val_split * len(dataset))
  test_size = int(test_split * len(dataset))
  train_size = len(dataset) - val_size - test_size

  train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size])

  collate_fn = lambda x: x
  train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
  if val_size != 0:
    val_loader = DataLoader(
      val_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
  else:
    val_loader = None
  if test_size != 0:
    test_loader = DataLoader(
      test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
  else:
    test_loader = None

  return train_loader, val_loader, test_loader

def get_vq_tokens_dataloaders(root_dir, batch_size=1, include_audio=True,
                    val_split=0.05, test_split=0.1, shuffle=True):
  dataset = OsuTokensDataset(root_dir, include_audio=include_audio)
  # Split dataset into train, val, and test
  val_size = int(val_split * len(dataset))
  test_size = int(test_split * len(dataset))
  train_size = len(dataset) - val_size - test_size

  train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size])

  collate_fn = lambda x: x
  train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
  if val_size != 0:
    val_loader = DataLoader(
      val_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
  else:
    val_loader = None
  if test_size != 0:
    test_loader = DataLoader(
      test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
  else:
    test_loader = None

  return train_loader, val_loader, test_loader


def load_token_data(path):
  """
  Loads token data from mapid.txt file
  
  Args:
    path: Path to the {mapid}.txt file.
  Returns:
    a list of token data
  """

  with open(path, 'r', encoding='utf8') as f:
    data = f.read()

  lines = data.split('\n')
  tokens = []
  for line in lines:
    if len(line) != 0:
      t = line.split(' ')
      tokens.append(t)

  return tokens

def sample_hitobjects(hit_objects,
    n_hit_objects=8):
  modified_hit_objects = []
  last_time = 0
  for hit_obj in hit_objects:
    s_arr = hit_obj.split(',')
    t = int(s_arr[2])
    if (t - last_time) >= 10000:
      last_time = t
      continue
    s_arr[2] = str(t - last_time)
    modified_hit_objects.append(','.join(s_arr))
    last_time = t
  
  start_idx = np.random.randint(0, max(1, len(modified_hit_objects) - n_hit_objects))
  selected_hit_objects = modified_hit_objects[start_idx:start_idx + n_hit_objects]

  return selected_hit_objects