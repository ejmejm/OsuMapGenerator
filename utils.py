import argparse
import os
import pathlib
import yaml

def load_config(path=None):
  if path is None:
    path = os.path.join(
      pathlib.Path(__file__).parent.resolve(),
      'config.yaml')
      
  with open(path, 'r') as f:
    return yaml.load(f, Loader=yaml.FullLoader)

def parse_args():
  # Create arguments
  parser = argparse.ArgumentParser()

  ### General ###
  
  parser.add_argument('--config', type=str, default='configs/default_config.yaml')

  ### Vocab Creation ###

  parser.add_argument('--early_stop', type=int, default=-1) # For vocab creation

  ### Generation ###
  
  parser.add_argument('--map_ids', type=str, nargs='*', default=None)
  parser.add_argument('--n_maps', type=int, default=1)
  # parser.add_argument('--n_variations', type=int, default=1)

  args = parser.parse_args()
  return args

def log(data, config):
  if config['use_wandb']:
    try:
      wandb.log(data)
    except NameError:
      import wandb
      wandb.log(data)