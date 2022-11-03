from decimal import Decimal, getcontext
import os
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

VERSION_PATTERN = r'osu file format v(\d+)(?://.*)?$'
METADATA_ENTRY_PATTERN = r'^([a-zA-Z]+):(.+?)(?://.*)?$'
TIMING_POINT_PATTERN = r'^([0-9,.-]+)(?://.*)?$'
TIMING_POINT_PATTERN = r'^([0-9,.-]+)(?://.*)?$'
HIT_OBJECT_PATTERN = r'^(.+?)(?://.*)?$'
HIT_OBJECT_START_TOKEN = '<Start>'
HIT_OBJECT_END_TOKEN = '<End>'


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
def get_dataloaders(root_dir, batch_size=1, include_audio=True,
                    val_split=0.05, test_split=0.1, shuffle=True):
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
  val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
  test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

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
  if start_idx == 0:
    selected_hit_objects.insert(0, HIT_OBJECT_START_TOKEN)
  if start_idx + n_hit_objects == len(hit_objects):
    selected_hit_objects.append(HIT_OBJECT_END_TOKEN)
  selected_time_points = time_points

  # TODO: Add a way to sample a portion of audio data
  selected_audio = []

  return selected_metadata, selected_time_points, \
         selected_hit_objects, selected_audio

def round_str(string, n=3):
  """Rounds a string-represented number to n decimal places"""
  return '{:f}'.format(round(Decimal(string), n).normalize())

def format_time_points(time_points):
  """
  Takes only necessary info from time points, truncates rounds long decimals,
  and splits into inhereted and uninhereted time points.
  """
  new_time_points = []
  slider_changes = []
  last_entry_type = None
  for tp in time_points:
    tp_data = tp.split(',')
    inhereted = not int(tp_data[6]) if len(tp_data) > 6 else False
    beat_len_str = round_str(tp_data[1])
    if inhereted:
      # Check if this slider change is just repeat data, and skip if so
      repeat = \
        last_entry_type == 'slider_change' \
        and slider_changes[-1].split(',')[1] == beat_len_str

      repeat = repeat or (
        last_entry_type == 'time_point' \
        and beat_len_str == '-100')

      if not repeat:
        slider_changes.append(f'{round_str(tp_data[0])},{beat_len_str}')
        last_entry_type = 'slider_change'
    else:
      new_time_points.append(f'{round_str(tp_data[0])},{beat_len_str},{tp_data[2]}')
      last_entry_type = 'time_point'

  return new_time_points, slider_changes

def get_time_points_in_range(time_points, hit_objects, slider_changes=None):
  """
  Returns a list of time points and slider changes that are in the range of the hit objects.
  """
  slider_changes = slider_changes or []
  if hit_objects[0] == HIT_OBJECT_START_TOKEN:
    start_time = 0
  else:
    start_time = int(hit_objects[0].split(',')[2])

  if hit_objects[-1] == HIT_OBJECT_END_TOKEN:
    end_time = 9999999
  else:    
    end_time = int(hit_objects[-1].split(',')[2])

  selected_time_points = []
  for tp in reversed(time_points):
    tp_time = float(tp.split(',')[0])
    if tp_time > end_time:
      continue
    selected_time_points.append(tp)
    if tp_time <= start_time:
      break
  selected_time_points = selected_time_points[::-1]

  if len(selected_time_points) == 0:
    selected_time_points = time_points[0]

  first_time_step = float(selected_time_points[0].split(',')[0])

  selected_slider_changes = []
  for sc in slider_changes:
    sc_time = float(sc.split(',')[0])
    if sc_time >= first_time_step:
      if sc_time <= end_time:
        selected_slider_changes.append(sc)
      else:
        break

  return selected_time_points, selected_slider_changes

def format_hit_objects(hit_objects):
  """
  Takes only necessary info from hit objects.
  """
  new_hit_objects = []
  for ho in hit_objects:
    if ho.startswith('<'):
      new_hit_objects.append(ho)
    else:
      ho_data = ho.split(',')
      # TODO: Bring back extra params and sliders
      new_hit_objects.append(','.join(ho_data[:4] + ['0']))
  return new_hit_objects

def format_training_data(metadata, time_points, hit_objects, audio_data):
  prior_str = '<Metadata>'
  for key, value in metadata.items():
    prior_str += f'<{key}>{value}'

  f_time_points, slider_changes = format_time_points(time_points)
  f_time_points, slider_changes = get_time_points_in_range(
    f_time_points, hit_objects, slider_changes)

  prior_str += '<TimePoints>'
  for tp in f_time_points:
    prior_str += f'<TimePointSplit>{tp}'

  if len(slider_changes) > 0:
    prior_str += '<SliderChanges>'
    for sc in slider_changes:
      prior_str += f'<SliderChange>{sc}'

  if audio_data:
    prior_str += '<Audio>'
    for audio in audio_data:
      prior_str += f'<AudioSplit>{audio}'

  pred_str = ''
  f_hit_objects = format_hit_objects(hit_objects)
  for ho in f_hit_objects:
    if ho in ['<s>', '</s>']:
      pred_str += ho
    else:
      pred_str += f'<HitObject>{ho}'
  # print('='*100)
  # print(prior_str)
  # print('-'*50)
  # print(pred_str)
  return prior_str, pred_str


if __name__ == '__main__':
  # dataset = OsuDataset('../data/formatted_beatmaps/')
  # print(len(dataset))
  # print(dataset[5])

  train_loader, val_loader, test_loader = get_dataloaders('../data/formatted_beatmaps/', batch_size=2)
  print(len(train_loader))
  full_data = next(iter(train_loader))
  print(full_data)

  sampled_data = sample_from_map(*full_data[1])
  print(sampled_data)