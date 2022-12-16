import re
import os
import pandas as pd
from evaluation.pattern_detect import pattern_detect
from evaluation.difficulty import anaylize_difficulty
from evaluation.compression_dist import time_diff, compression_dist
from utils import load_config, log, parse_args


VERSION_PATTERN = r'osu file format v(\d+)(?://.*)?$'
METADATA_ENTRY_PATTERN = r'^([a-zA-Z]+):(.+?)(?://.*)?$'

ROOT_DIR = 'evaluation_data'

MODEL_TO_PATH = {
    "random": os.path.join(ROOT_DIR, 'random'),
    "ours": os.path.join(ROOT_DIR, 'ours'),
    "osumapper": os.path.join(ROOT_DIR, 'osumapper'),
    "human": os.path.join(ROOT_DIR, 'maps')
}

GROUND_TRUTH_PATH = os.path.join(ROOT_DIR, 'maps')

FILE_NAME_PATTERN = "{}.osu"
MAX_FILE_INDEX = 1

def evaluation_stub1(osu_str):
    return {"stub1": 1.0}

def evaluation_stub2(osu_str):
    return {"stub2": 2.0}

EVALUATION_METHOD = {
    "difficulty": anaylize_difficulty,
    "patterns": pattern_detect
}

EVALUATION_METHOD_WITH_GT = {
    "compression distance": compression_dist,
    "time diff": time_diff
}

CSV_COLUMNS = ["file_name", "model_name"]


def read_file(path):
    with open(path, 'r', encoding='utf8') as f:
        data = f.read()
        return data

def get_metadata(data):
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
    return metadata

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)


    all_metric_list = []
    META_DATA_COLUMNS = set()
    METRICS_COLUMNS = set()

    mapping = pd.read_csv(
      os.path.join(ROOT_DIR, 'song_mapping.csv'), index_col=0)
    mapping = mapping.to_dict()['song']
    map_list = list(mapping.keys())
    # map_list = ['14555.osu']

    for model_name, method_osu_path in MODEL_TO_PATH.items():
        for file_name in map_list:
            osu_data = read_file(os.path.join(method_osu_path, file_name))
            gt_osu_data = read_file(os.path.join(GROUND_TRUTH_PATH, file_name))
            meta_data_map = get_metadata(osu_data)
            all_metric_map = {}
            for metric_name, method_name in EVALUATION_METHOD.items():
                one_metric_map = method_name(osu_data)
                all_metric_map.update(one_metric_map)
                METRICS_COLUMNS.update(one_metric_map.keys())

            for metric_name, method_name in EVALUATION_METHOD_WITH_GT.items():
                one_metric_map = method_name(osu_data, gt_osu_data)
                all_metric_map.update(one_metric_map)
                METRICS_COLUMNS.update(one_metric_map.keys())
            all_metric_map['file_name'] = file_name
            all_metric_map.update(meta_data_map)
            all_metric_map['model_name'] = model_name
            all_metric_list.append(all_metric_map)
            META_DATA_COLUMNS.update(meta_data_map.keys())
    
    CSV_COLUMNS.extend(METRICS_COLUMNS)
    CSV_COLUMNS.extend(META_DATA_COLUMNS)
    df = pd.DataFrame.from_dict(all_metric_list)
    df.to_csv('output/evaluation.csv', index = False, header=True, columns=CSV_COLUMNS)