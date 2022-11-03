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

def log(data, config):
  if config['use_wandb']:
    try:
      wandb.log(data)
    except NameError:
      import wandb
      wandb.log(data)