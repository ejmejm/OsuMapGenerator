from decimal import Decimal, getcontext
import os
import re
import warnings

from preprocessing.audio_processing import prepare_audio_tensor, process_song, load_audio, audio_to_np
from preprocessing.text_processing import get_text_preprocessor, prepare_tensor_seqs
from utils import load_config
from pydub.exceptions import CouldntDecodeError

from essentia.standard import MonoLoader
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
BREAK_TOKEN = '<Break>'
AUDIO_SEGMENT_TOKEN = '<AudioSegment>'
AUDIO_PLACEHOLDER_TOKEN = '<AudioPlaceholder>'

DEFAULT_METADATA = set([
  'DistanceSpacing', # 'AudioLeadIn', 'Countdown', 'CountdownOffset', 
  'BeatDivisor', 'GridSize', 'CircleSize', 'OverallDifficulty', 'ApproachRate',
  'SliderMultiplier', 'SliderTickRate', 'HPDrainRate', 'FormatVersion'
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

class OsuDataset(Dataset):
  def __init__(self, config, map_ids=None, preprocess=False):
    self.config = config
    self.root_dir = config['beatmap_path']
    self.sample_rate = config['sample_rate']
    self.preprocess = preprocess
    self.include_audio = config['include_audio']
    self.preprocess_text = get_text_preprocessor(config)[0]

    self.map_dir = os.path.join(self.root_dir, 'maps')
    self.audio_dir = os.path.join(self.root_dir, 'songs')
  
    # Mapping of map_id to song_id
    self.mapping = pd.read_csv(
      os.path.join(self.root_dir, 'song_mapping.csv'), index_col=0)
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
      audio_path = os.path.join(self.audio_dir, self.mapping[map_id])
      # TODO: Delete beatmaps with bad audio in preprocessing
      # Curretly takes ~200-1000ms to load a song
      audio_data = MonoLoader(filename=audio_path, sampleRate=self.sample_rate)()
      # print(audio_data)
      # RuntimeError: Error while configuring MelBands: TriangularBands: the number of spectrum bins is insufficient for the specified number of triangular bands. Use zero padding to increase the number of FFT bins.
    else:
      audio_data = None

    if self.preprocess:
      out = map_to_train_data(
        (metadata, time_points, hit_objects, audio_data),
        self.preprocess_text, self.config, device='cpu')
      return out

    return metadata, time_points, hit_objects, audio_data

# Get dataloaders for training test and validation
def get_dataloaders(config, preprocess=False, shuffle=True):
  # If preprocess_text is passed in, all the preprocessing is done in the dataloader
  if preprocess:
    collate_fn = map_to_train_collate
  else:
    collate_fn = lambda x: x

  dataset = OsuDataset(config, preprocess=preprocess)

  # Split dataset into train, val, and test
  val_size = int(config['val_split'] * len(dataset))
  test_size = int(config['test_split'] * len(dataset))
  train_size = len(dataset) - val_size - test_size

  train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size])

  if config['n_load_workers'] > 0:
    torch.multiprocessing.set_start_method('spawn')

  train_loader = DataLoader(
    train_dataset, batch_size=config['batch_size'], shuffle=shuffle, collate_fn=collate_fn,
    num_workers=config['n_load_workers'])
  val_loader = DataLoader(
    val_dataset, batch_size=config['batch_size'], shuffle=shuffle, collate_fn=collate_fn,
    num_workers=config['n_load_workers'])
  test_loader = DataLoader(
    test_dataset, batch_size=config['batch_size'], shuffle=shuffle, collate_fn=collate_fn,
    num_workers=config['n_load_workers'])

  return train_loader, val_loader, test_loader


# Samples a portion of data from and entire set of beatmap data
# Target metadata items, all time_points, prior hit_objects,
# new hit_objects, prior audio, new audio
def sample_from_map(
    metadata, time_points, hit_objects, audio_data,
    config, target_metadata=DEFAULT_METADATA):

  selected_metadata = {}
  for key in target_metadata:
    value = metadata.get(key)
    if value is not None:
      selected_metadata[key] = value

  start_idx = np.random.randint(0, max(1, len(hit_objects) - config['max_hit_objects']))
  selected_hit_objects = hit_objects[start_idx:start_idx + config['max_hit_objects']]
  if start_idx == 0:
    selected_hit_objects.insert(0, HIT_OBJECT_START_TOKEN)
    start_time = 0
  else:
    start_time = int(selected_hit_objects[0].split(',')[2])

  if start_idx + config['max_hit_objects'] == len(hit_objects):
    selected_hit_objects.append(HIT_OBJECT_END_TOKEN)

  selected_time_points = time_points

  if audio_data is not None:
    # TODO: Figure out what the end time for the audio should be
    # selected_audio = audio_data[start_time:]
    selected_audio = audio_data
  else:
    # selected_audio = [[(np.zeros(80) if ((hit[2]//config["segment_length"] + i) < 0 or (hit[2]//config["segment_length"] + i) >= len(audio_data)) else audio_data[hit[2]//config["segment_length"] + i]) for i in range(-config["prev_audio_segments_per_hitobject"], config["next_audio_segments_per_hitobject"] + 1)] for hit in selected_hit_objects]
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
    selected_time_points = [time_points[0]]

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
      new_hit_objects.append(','.join(ho_data[:3] + ['1', '0']))
  return new_hit_objects

def format_time_diff(time_diff, beat_len, precision=5):
  """
  Converts a time diff to a relative string format.

  Format of time: <1>:<1/2><1/4><1/8><1/16><1/32> where each number is
    the fraction of the beat length.
  For example, 5:0100 means 5 + 1/4 beats.
  The above can also be truncated to 5:01.

  Returns the time diff as a string and the rounding error remainder.
  """
  min_diff = beat_len / (2 ** precision)
  time_diff += min_diff / 2 # For rounding
  remainder = time_diff % beat_len

  time_diff_str = str(int(time_diff / beat_len))

  if remainder < min_diff:
    return time_diff_str, remainder
  
  time_diff_str += ':'
  for i in range(precision):
    if remainder < min_diff:
      break
    elif remainder >= beat_len / (2 ** (i + 1)):
      time_diff_str += '1'
      remainder -= beat_len / (2 ** (i + 1))
    else:
      time_diff_str += '0'

  if time_diff_str == '0' and time_diff > min_diff / 2:
    warnings.warn(
      f'Time diff {time_diff} is too small to ' + \
      f'be represented with beat length {beat_len}')

  return time_diff_str, remainder - min_diff / 2

def convert_to_relative_time(hit_objects, time_points):
  """
  Converts hit objects and time points to relative time.

  Format of time: <1>:<1/2><1/4><1/8><1/16><1/32> where each number is
    the fraction of the beat length.
  For example, 5:0100 means 5 + 1/4 beats.
  The above can also be truncated to 5:01.
  """
  hit_object_times = []
  for ho in hit_objects:
    if ho == HIT_OBJECT_START_TOKEN:
      hit_object_times.append(0)
    elif ho == HIT_OBJECT_END_TOKEN:
      hit_object_times.append(hit_object_times[-1])
    else:
      hit_object_times.append(int(ho.split(',')[2]))

  time_point_times = [float(tp.split(',')[0]) for tp in time_points]
  beat_lengths = [float(tp.split(',')[1]) for tp in time_points]

  time_point_idx = 0
  ho_beat_lengths = []
  new_hit_objects = []
  for i in range(1, len(hit_objects)):
    if hit_objects[i] in (HIT_OBJECT_START_TOKEN, HIT_OBJECT_END_TOKEN):
      new_hit_objects.append(hit_objects[i])
      continue

    # Key time_point_idx at the last time point before the hit object
    is_last_time_point = time_point_idx == len(time_point_times) - 1
    while not is_last_time_point \
          and hit_object_times[i] >= time_point_times[time_point_idx + 1]:
      time_point_idx += 1
      is_last_time_point = time_point_idx == len(time_point_times) - 1
    
    # Compute the relative time string
    beat_length = beat_lengths[time_point_idx]
    last_ho_time = hit_object_times[i - 1]
    ho_time = hit_object_times[i]
    time_diff = ho_time - last_ho_time
    time_diff_str = format_time_diff(time_diff, beat_length)[0]

    # Create the new hit object with the relative time string
    hit_object_data = hit_objects[i].split(',')
    new_hit_objects.append(','.join(hit_object_data[:2] + [time_diff_str] + hit_object_data[3:]))
    ho_beat_lengths.append(beat_length)

  return new_hit_objects, ho_beat_lengths

def get_relative_time_beats(hit_object):
  """
  Gets the number of beats in a relative time string.
  """
  beats = float(hit_object[0])
  if len(hit_object) <= 1 or hit_object[1] != ':':
    return beats

  for i in range(2, len(hit_object)):
    beats += 1 / (2 ** (i - 1))
  return beats

def get_hit_object_time(hit_object, relative=False):
  """When relative gets the number of beats, otherwise gets the time."""
  if hit_object == HIT_OBJECT_START_TOKEN:
    return 0
  elif hit_object == HIT_OBJECT_END_TOKEN:
    return -1
  
  time_str = hit_object.split(',')[2]
  if relative:
    return get_relative_time_beats(time_str)
  return float(time_str)

def add_hit_object_breaks(hit_objects, break_length):
  """
  Add breaks to hit objects when there are long delays with no hit objects.
  Currently only works with relative time.
  """
  new_hit_objects = [hit_objects[0]]

  for i in range(1, len(hit_objects)):
    beats = get_hit_object_time(hit_objects[i], relative=True)
    if beats > break_length:
      # Add enough breaks to fill the gap
      n_breaks = int(np.ceil((beats / break_length) - 1))
      new_hit_objects.extend([BREAK_TOKEN for _ in range(n_breaks)])
      new_beats = beats - n_breaks * break_length
      time_str = format_time_diff(new_beats, 1)[0]

      ho_data = hit_objects[i].split(',')
      new_hit_objects.append(','.join(ho_data[:2] + [time_str] + ho_data[3:]))
    else:
      new_hit_objects.append(hit_objects[i])

  return new_hit_objects

def format_training_data(
  metadata, time_points, hit_objects, audio_data, config):
  prior_str = '<Metadata>'
  for key, value in metadata.items():
    if key in DEFAULT_METADATA:
      prior_str += f'<{key}>{value}'

  f_time_points, slider_changes = format_time_points(time_points)

  # if config['relative_timing']:
  #   hit_objects = hit_objects[1:]

  orig_hit_objects = hit_objects
  f_time_points, slider_changes = get_time_points_in_range(
    f_time_points, hit_objects, slider_changes)
  r_hit_objects, bpms = convert_to_relative_time(hit_objects, f_time_points)

  if config['relative_timing']:
    hit_objects = r_hit_objects
    if config['break_length'] > 0:
      hit_objects = add_hit_object_breaks(hit_objects, config['break_length'])

  prior_str += '<TimePoints>'
  for tp in f_time_points:
    prior_str += f'<TimePointSplit>{tp}'

  if len(slider_changes) > 0:
    prior_str += '<SliderChanges>'
    for sc in slider_changes:
      prior_str += f'<SliderChange>{sc}'

  pred_str = ''
  f_hit_objects = format_hit_objects(hit_objects)
  if config.get('max_hit_objects'):
    f_hit_objects = f_hit_objects[:config.get('max_hit_objects')]

  if audio_data != [] and audio_data is not None:
    audio_segments = []
    orig_ho_idx = 0
    prev_ho_time = 0
    for i in range(len(f_hit_objects)):
      if f_hit_objects[i] == BREAK_TOKEN:
        bpm = bpms[orig_ho_idx - 1]
        ho_time = prev_ho_time + config['break_length'] * bpms[orig_ho_idx - 1]
      else:
        bpm = bpms[orig_ho_idx]
        ho_time = get_hit_object_time(orig_hit_objects[orig_ho_idx], relative=False)
        orig_ho_idx += 1
        if ho_time == -1:
          continue

      start_idx = int(ho_time * config['sample_rate'] / 1000)
      end_idx = int((ho_time + bpm * config['break_length']) * config['sample_rate'] / 1000)
      segment = audio_data[start_idx:end_idx]

      if end_idx > len(audio_data):
        segment = np.concatenate([segment, np.zeros((end_idx - len(audio_data),))])
      mel_bands = audio_to_np(
        segment, config['segments_per_beat'] * config['break_length'],
        n_mel_bands=config['n_mel_bands'], sample_rate=config['sample_rate'])
      audio_segments.append(mel_bands)

      prev_ho_time = ho_time
    audio_segments = np.stack(audio_segments)
  else:
    audio_segments = None

  n_audio_tokens = 0 if audio_segments is None else config['audio_tokens_per_segment']
  for ho in f_hit_objects:
    if ho in [HIT_OBJECT_START_TOKEN, HIT_OBJECT_END_TOKEN]:
      pred_str += ho
    else:
      pred_str += f'<HitObject>{ho}'
    
    if n_audio_tokens > 0 and ho != HIT_OBJECT_END_TOKEN:
      pred_str += '<AudioSegment>'
      pred_str += '<AudioPlaceholder>' * n_audio_tokens

  return prior_str, pred_str, audio_segments

def map_to_train_data(map, preprocess_text, config, device=None):
  batch_samples = [sample_from_map(*m, config) for m in [map]]
  training_samples = [format_training_data(*m, config) for m in batch_samples]

  src, tgt, audio_segments = zip(*training_samples)
  # Convert text to numerical tensors with padding and corresponding masks
  src_tensor, tgt_tensor, src_mask, tgt_mask = \
    prepare_tensor_seqs(src, tgt, preprocess_text, config, device=device)
  # Split the tgt tensor into the input and actual target
  target = tgt_tensor[1:]
  tgt_tensor = tgt_tensor[:-1]
  tgt_mask = tgt_mask[:-1, :-1]

  if audio_segments is not None and config['include_audio']:
    audio_segments = prepare_audio_tensor(
      audio_segments, config, device=device)
    audio_token_id = preprocess_text(AUDIO_PLACEHOLDER_TOKEN)[0]
    audio_idxs = tgt_tensor == audio_token_id
    audio_idxs = audio_idxs.to(device or config['device'])
  else:
    audio_segments = None
    audio_idxs = None

  return src_tensor, tgt_tensor, src_mask, tgt_mask, \
         audio_segments, audio_idxs, target

def map_to_train_collate(batch):
  cat_dims = [1, 1, None, None, 0, 1, 1] # Dimensions to concatenate
  out_data = []
  for tensors, dim in zip(zip(*batch), cat_dims):
    if dim is None:
      out_data.append(tensors[0])
    else:
      out_data.append(torch.cat(tensors, dim))

  return tuple(out_data)

if __name__ == '__main__':
  # dataset = OsuDataset('../data/formatted_beatmaps/')
  # print(len(dataset))
  # print(dataset[5])

  train_loader, val_loader, test_loader = get_dataloaders('./data/formatted_beatmaps/', batch_size=2)
  # print(len(train_loader))
  full_data = next(iter(train_loader))
  # print(full_data)

  sampled_data = sample_from_map(*full_data[1], n_hit_objects=50)
  # print(sampled_data)

  formatted_data = format_training_data(*sampled_data, relative_timing=True, break_length=4)
  print(formatted_data[1].replace('<HitObject>', '\n'))
