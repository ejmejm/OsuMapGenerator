import os
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from utils import load_config

VERSION_PATTERN = r'osu file format v(\d+)(?://.*)?$'
METADATA_ENTRY_PATTERN = r'^([a-zA-Z]+):(.+?)(?://.*)?$'
TIMING_POINT_PATTERN = r'^([0-9,.]+)(?://.*)?$'
HIT_OBJECT_PATTERN = r'^(.+?)(?://.*)?$'



DEFAULT_METADATA = set([
  'DistanceSpacing', # 'AudioLeadIn', 'Countdown', 'CountdownOffset', 
  'BeatDivisor', 'GridSize', 'CircleSize', 'OverallDifficulty', 'ApproachRate',
  'SliderMultiplier', 'SliderTickRate', 'HPDrainRate'
])


def load_beatmap_data(path):
  """
  Loads beatmap data from a .osu file.
  
  Args:
    path: Path to the .osu file.
  Returns:
    A dictionary and two lists containing metatdata, timing points, and hit objects.
  """

  with open(path, 'r', encoding='utf8') as f:
    data = f.read()

  lines = data.split('\n')
  metadata = {}
  timing_points = []
  hit_objects = []

  # Get the osu file format version
  result = re.search(VERSION_PATTERN, lines[0])
  groups = result.groups() if result else tuple([None])
  metadata['FormatVersion'] = groups[0]

  ### Parse general metadata ###

  curr_heading = ''
  for line_idx, line in enumerate(lines):
    line = line.strip()

    # Stop metadata parsing when we reach the hitobjects
    if line == '[TimingPoints]':
      break

    result = re.search(METADATA_ENTRY_PATTERN, line)
    groups = result.groups() if result else tuple()

    # This is a metadata entry
    if len(groups) >= 2:
      key, value = groups[0].strip(), groups[1].strip()
      metadata[key] = value

    # Capture section headings
    elif line.startswith('['):
      curr_heading = line[1:-1]

    # Skip comments and emtpy lines
    elif line.startswith('//') or line == '':
      continue

  ### Parse timing points and hit objects ###

  for line_idx, line in enumerate(lines, start=line_idx):
    line = line.strip()

    # Capture section headings
    if line.startswith('['):
      curr_heading = line[1:-1]
      continue

    # Skip comments and emtpy lines
    elif line.startswith('//') or line == '':
      continue

    # Check for timing points
    if curr_heading == 'TimingPoints':
      result = re.search(TIMING_POINT_PATTERN, line)
      groups = result.groups() if result else tuple()
      if len(groups) >= 1:
        timing_points.append(groups[0].strip())

    # Check for hit objects
    elif curr_heading == 'HitObjects':
      result = re.search(HIT_OBJECT_PATTERN, line)
      groups = result.groups() if result else tuple()
      if len(groups) >= 1:
        hit_objects.append(groups[0].strip())

  return metadata, timing_points, hit_objects

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
  for line in enumerate(lines):
    tokens.append(line.strip().split(' '))

  return tokens

def load_beatmap_token_data(path):
  """
  Loads beatmap data from a .osu file.
  
  Args:
    path: Path to the .osu file.
  Returns:
    A dictionary and two lists containing metatdata, timing point tokens, and hit object tokens.
  """

  with open(path, 'r', encoding='utf8') as f:
    data = f.read()

  lines = data.split('\n')
  metadata = {}
  timing_points_token = []
  hit_objects_token = []

  return metadata, timing_points_token, hit_objects_token

# TODO: Add breaks to dataset
class OsuDataset(Dataset):
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
    metadata, time_points, hit_objects = load_beatmap_data(
      os.path.join(self.map_dir, map_id))
    if self.include_audio:
      # TODO: Add audio loading here
      audio_data = []
    else:
      audio_data = []

    return metadata, time_points, hit_objects, audio_data

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

    tokens = load_token_data(os.path.join(self.map_dir, map_id, ".txt"))

    return metadata, time_points, tokens, audio_data


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

# Samples a portion of data from and entire set of beatmap data
# Target metadata items, all time_points, prior hit_objects,
# new hit_objects, prior audio, new audio
def sample_split_from_map(
    metadata, time_points, hit_objects, audio_data,
    n_prior_ho=5, n_new_ho=5, prior_audio_secs=1, new_audio_secs=2,
    target_metadata=DEFAULT_METADATA):

  selected_metadata = {}
  for key in target_metadata:
    value = metadata.get(key)
    if value is not None:
      selected_metadata[key] = value

  start_idx = np.random.randint(0, len(hit_objects))
  prior_ho = hit_objects[max(0, start_idx - n_prior_ho):start_idx]
  new_ho = hit_objects[start_idx:start_idx + n_new_ho]

  selected_time_points = time_points

  # TODO: Add a way to sample a portion of audio data
  prior_audio = []
  new_audio = []

  return selected_metadata, selected_time_points, prior_ho, \
         new_ho, prior_audio, new_audio

def sample_from_map(
    metadata, time_points, hit_objects, audio_data,
    n_hit_objects=10, audio_secs=5, target_metadata=DEFAULT_METADATA):

  selected_metadata = {}
  for key in target_metadata:
    value = metadata.get(key)
    if value is not None:
      selected_metadata[key] = value

  start_idx = np.random.randint(0, max(1, len(hit_objects) - n_hit_objects))
  selected_hit_objects = hit_objects[start_idx:start_idx + n_hit_objects]
  selected_time_points = time_points

  # TODO: Add a way to sample a portion of audio data
  selected_audio = []

  return selected_metadata, selected_time_points, \
         selected_hit_objects, selected_audio

def sample_hitobjects(hit_objects,
    n_hit_objects=8):

  start_idx = np.random.randint(0, max(1, len(hit_objects) - n_hit_objects))
  selected_hit_objects = hit_objects[start_idx:start_idx + n_hit_objects]

  return selected_hit_objects

def format_training_data(metadata, time_points, hit_objects, audio_data):
  prior_str = ''
  if metadata:
    prior_str = '<Metadata>'
    for key, value in metadata.items():
      prior_str += f'<{key}>{value}'

  if time_points:
    prior_str += '<TimePoints>'
    for tp in time_points:
      prior_str += f'<TimePointSplit>{tp}'

  if audio_data:
    prior_str += '<Audio>'
    for audio in audio_data:
      prior_str += f'<AudioSplit>{audio}'

  if hit_objects:
    pred_str = ''
    for ho in hit_objects:
      pred_str += f'<HitObject>{ho}'
    pred_str += '<HitObject>'

  return prior_str, pred_str


if __name__ == '__main__':
  # dataset = OsuDataset('../data/formatted_beatmaps/')
  # print(len(dataset))
  # print(dataset[5])
  config = load_config()
  train_loader, val_loader, test_loader = get_dataloaders(config, '../data/formatted_beatmaps/', batch_size=2)
  print(len(train_loader))
  full_data = next(iter(train_loader))
  print(full_data)

  sampled_data = sample_from_map(*full_data[1])
  print(sampled_data)