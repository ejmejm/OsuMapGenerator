from decimal import Decimal, getcontext
import os
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import asyncio

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

async def read_one(map_dir, map_id, map_ids):
  p = os.path.join(map_dir, map_id)
  with open(p, 'r', encoding='utf8') as f:
    data = f.read()

    lines = data.split('\n')
    # Get the osu file format version
    result = re.search(VERSION_PATTERN, lines[0])
    groups = result.groups() if result else tuple([None])
    if groups[0] == '14':
        map_ids.add(map_id)

async def filterV14(root_dir):
  map_dir = os.path.join(root_dir, 'maps')
  mapdf = pd.read_csv(
    os.path.join(root_dir, 'song_mapping.csv'), index_col=0)
  mapping = mapdf.to_dict()['song']
  map_list = list(mapping.keys())
  map_ids = set()
  tasks = [asyncio.create_task(read_one(map_dir, map_id, map_ids)) for map_id in map_list]
  for task in tasks:
    await task
    

  new_dic = {k: v for k, v in mapping.items() if map_ids.__contains__(k)}
  with open(os.path.join(root_dir, 'bksong_mapping2.csv'), 'w', encoding='utf8') as f:
    for k, v in new_dic.items():
        f.write(k)
        f.write(',')
        f.write(v)
        f.write('\n')
  new_df = pd.DataFrame(new_dic.items())
  new_df.to_csv(os.path.join(root_dir, 'bksong_mapping.csv'))
  return map_ids

if __name__ == '__main__':
  asyncio.run(filterV14("/mnt/d/course_project/OsuMapGenerator/data/newdata"))