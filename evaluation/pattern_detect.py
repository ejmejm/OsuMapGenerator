import os
import re
from utils import load_config
from preprocessing.data_loading import load_beatmap_data
import numpy as np

VERSION_PATTERN = r'osu file format v(\d+)(?://.*)?$'
METADATA_ENTRY_PATTERN = r'^([a-zA-Z]+):(.+?)(?://.*)?$'
TIMING_POINT_PATTERN = r'^([0-9,.-]+)(?://.*)?$'
HIT_OBJECT_PATTERN = r'^(.+?)(?://.*)?$'

DEFAULT_METADATA = set([
    'DistanceSpacing', # 'AudioLeadIn', 'Countdown', 'CountdownOffset', 
    'BeatDivisor', 'GridSize', 'CircleSize', 'OverallDifficulty', 'ApproachRate',
    'SliderMultiplier', 'SliderTickRate', 'HPDrainRate'
])

# Threshold parameters
jump = 100
near = 40
backandfor = 40


def get_location_diff(data1, data2):
    x1 = int(data1[0])
    x2 = int(data2[0])
    y1 = int(data1[1])
    y2 = int(data2[1])
    xdiff = abs(x1-x2)
    ydiff = abs(y1-y2)
    return [xdiff,ydiff]

def get_distance(loca_diff):
    return (loca_diff[0] ** 2 + loca_diff[1] ** 2) ** 0.5

def get_time_diff(data1, data2):
    return abs(int(data1[2]) - int(data2[2]))

def pattern_jump(data1, data2, beat):
    loca_diff = get_location_diff(data1, data2)
    distance = get_distance(loca_diff)
    time_diff = get_time_diff(data1, data2)
    if distance > jump and time_diff < beat:
        return 1
    return 0

def pattern_doubletap(data1, data2, beat):
    x = get_location_diff(data1, data2)[0]
    y = get_location_diff(data1, data2)[1]
    time_diff = get_time_diff(data1, data2)
    if x == 0 and y == 0 and time_diff <= (beat/2):
        return 1
    return 0

def pattern_trippletap(data1, data2, data3, beat):
    if pattern_doubletap(data1, data2, beat) == 1 and pattern_doubletap(data2, data3, beat) == 1:
        return 1
    return 0

def pattern_backandfor(data1, data2, data3):
    loca_diff13 = get_location_diff(data1, data3)
    loca_diff23 = get_location_diff(data2, data3)
    distance13 = get_distance(loca_diff13)
    distance23 = get_distance(loca_diff23)

    if distance13 <= near and distance23 >= backandfor:
        return 1
    return 0

def pattern_stream(data1, data2, data3, data4, data5, beat):
    loca_diff12 = get_location_diff(data1, data2)
    loca_diff23 = get_location_diff(data2, data3)
    loca_diff34 = get_location_diff(data3, data4)
    loca_diff45 = get_location_diff(data4, data5)
    distance12 = get_distance(loca_diff12)
    distance23 = get_distance(loca_diff23)
    distance34 = get_distance(loca_diff34)
    distance45 = get_distance(loca_diff45)
    time_diff12 = get_time_diff(data1, data2)
    time_diff23 = get_time_diff(data2, data3)
    time_diff34 = get_time_diff(data3, data4)
    time_diff45 = get_time_diff(data4, data5)
    if max(distance12,distance23,distance34,distance45) <= near and max(time_diff12, time_diff23, time_diff34, time_diff45) <= beat:
        return 1
    return 0



def pattern_detect(data):
    """
    Loads beatmap data from a .osu file.
    
    Args:
        path: Path to the .osu file.
    Returns:
        A dictionary and two lists containing metatdata, timing points, and hit objects.
    """
    try:
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

            
        # Get beat
        beat_str = timing_points[0].split(',')[1]
        beat = int(beat_str.split('.')[0]) + 1

        # Detect pattern
        pjump = 0
        pdoubletap = 0
        ptrippletap = 0
        pbackandfor = 0
        pstream = 0

        length = len(hit_objects)
        for i in range(0,length-1):
            data1 = hit_objects[i].split(',')
            if (i+4) < length:
                data2 = hit_objects[i+1].split(',')
                data3 = hit_objects[i+2].split(',')
                data4 = hit_objects[i+3].split(',')
                data5 = hit_objects[i+4].split(',')
                if pattern_stream(data1, data2, data3, data4, data5, beat) == 1:
                    pstream += 1
                    i += 4
            if (i+2) < length:
                data2 = hit_objects[i+1].split(',')
                data3 = hit_objects[i+2].split(',')
                if pattern_backandfor(data1, data2, data3) == 1:
                    pbackandfor += 1
                    i += 2
                if pattern_trippletap(data1, data2, data3, beat) == 1:
                    ptrippletap += 1
                    i += 2

            if (i+1) < length:
                data2 = hit_objects[i+1].split(',')
                pjump += pattern_jump(data1, data2, beat)
                pdoubletap += pattern_doubletap(data1, data2, beat)

        pdensity = pjump + pdoubletap + ptrippletap + pbackandfor + pstream
        pvariations = (pjump != 0) + (pdoubletap != 0) + (ptrippletap != 0) + (pbackandfor != 0) + (pstream != 0)

        return {
            'Pattern-Jump': pjump, 
            'Pattern-doubletap': pdoubletap, 
            'Pattern-trippletap': ptrippletap, 
            'Pattern-backandfor': pbackandfor, 
            'Pattern-stream': pstream, 
            'Pattern Density': pdensity, 
            'Pattern variations': pvariations
            }
    except:
        print("error occurs in pattern detection")
        return  {
            'Pattern-Jump': np.nan, 
            'Pattern-doubletap': np.nan, 
            'Pattern-trippletap': np.nan, 
            'Pattern-backandfor': np.nan, 
            'Pattern-stream': np.nan, 
            'Pattern Density': np.nan, 
            'Pattern variations': np.nan
            }



if __name__ == '__main__':
    with open("data\\formatted_beatmaps\maps\\2.osu", 'r', encoding='utf8') as f:
        data = f.read()

    dict = pattern_detect(data)
    print(dict)