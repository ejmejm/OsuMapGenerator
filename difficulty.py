import numpy as np
import math
from preprocessing.data_loading import load_beatmap_data
from utils import parse_args

STAR_SCALING_FACTOR = 0.045
EXTREME_SCALING_FACTOR = 0.5
PLAYFIELD_WIDTH = 512

STRAIN_STEP = 400
DECAY_WEIGHT = 0.9

DECAY_BASE = [0.3, 0.15]


def anaylize_difficulty(map_path):
    meta, _, hitobjects = load_beatmap_data(map_path)
    hit_objects = [hit_object(hit) for hit in hitobjects]
    
    difficulty_circle_size = float(meta["CircleSize"])

    return run_all_rules(hit_objects, difficulty_circle_size), float(meta["OverallDifficulty"])

def run_all_rules(hit_objects, difficulty_circle_size):
    tp_hit_objects = []
    
    circle_radius = (PLAYFIELD_WIDTH / 16.0) * (1.0 - 0.7 * (difficulty_circle_size - 5.0) / 5.0)

    for hit_object in hit_objects:
        tp_hit_objects.append(tp_hit_object(hit_object, circle_radius))

    # tp_hit_objects.sort(lambda a,b: a.base_hit_object.start_time - b.base_hit_object.start_time)
    tp_hit_objects = sorted(tp_hit_objects, key=lambda x: x.base_hit_object.start_time, reverse=False)

    if(not calculate_strain_values(tp_hit_objects)):
        print("Could not compute strain values. Aborting difficulty calculation.")
        return
    
    speed_difficulty = calculate_difficulty(0, tp_hit_objects)
    aim_difficulty = calculate_difficulty(1, tp_hit_objects)

    speed_stars = math.sqrt(speed_difficulty) * STAR_SCALING_FACTOR
    aim_stars = math.sqrt(aim_difficulty) * STAR_SCALING_FACTOR

    star_rating = speed_stars + aim_stars + abs(speed_stars - aim_stars) * EXTREME_SCALING_FACTOR

    return star_rating


def calculate_strain_values(tp_hit_objects):
    if(len(tp_hit_objects) == 0):
        print("Can not compute difficulty of empty beatmap.")
        return False
    
    current_hit_object = tp_hit_objects[0]
    next_hit_object = None

    for i in range(1, len(tp_hit_objects)):
        next_hit_object = tp_hit_objects[i]
        next_hit_object.calculate_strains(current_hit_object)
        current_hit_object = next_hit_object

    return True


def calculate_difficulty(type, tp_hit_objects):
    highest_strains = []
    interval_end_time = STRAIN_STEP
    maximum_strain = 0

    previous_hit_object = None

    for hit_object in tp_hit_objects:
        while(hit_object.base_hit_object.start_time > interval_end_time):
            highest_strains.append(maximum_strain)

            if(previous_hit_object is None): maximum_strain = 0
            else:
                decay = pow(DECAY_BASE[type], (interval_end_time - previous_hit_object.base_hit_object.start_time) / 1000)
                maximum_strain = previous_hit_object.strains[type] * decay

            interval_end_time += STRAIN_STEP
        
        if(hit_object.strains[type] > maximum_strain):
            maximum_strain = hit_object.strains[type]

        previous_hit_object = hit_object
    
    difficulty = 0
    weight = 1
    
    # highest_strains.sort(lambda a,b : b.compareTo(a))
    highest_strains = sorted(highest_strains, reverse=True)

    for strain in highest_strains:
        difficulty += weight * strain
        weight *= DECAY_WEIGHT

    return difficulty


class hit_object: #TODO
    def __init__(self, string):
        feats = string.split(',') #640*480 x,y,time,type,hitSound,objectParams,hitSample
        self.position = np.array([int(feats[0]), int(feats[1])])
        self.type = "normal" if int(feats[3]) & 1 else "slider" if int(feats[3]) & 2 else "spinner"# if int(feats[3]) & 4 else "hold"
        self.start_time = int(feats[2])
        self.end_position = self.position #TODO check if this is right

        self.segment_count = int(feats[6]) if self.type == "slider" else 0
        self.length = round(float(feats[7])) if self.type == "slider" else 0

        # if(self.type == "slider"):
        #     self.segment_count = int(feats[6])
        #     self.length = int(feats[7])

    def position_at_time(self, time):
        return self.position#TODO

def vector_length(vector):
    return np.linalg.norm(vector)

def normalize_vector(vector):
    norm=np.linalg.norm(vector)
    if norm==0:
        norm=np.finfo(vector.dtype).eps
    return vector/norm

class tp_hit_object:
    def __init__(self, base_hit_object, circle_radius):
        LAZY_SLIDER_STEP_LENGTH = 1
        
        self.strains = [1, 1]
        self.base_hit_object = base_hit_object
        scaling_factor = 52.0 / circle_radius
        self.normalized_start_position = base_hit_object.position * scaling_factor
        self.normalized_end_position = self.base_hit_object.end_position * scaling_factor

        self.lazy_slider_length_first = 0
        self.lazy_slider_length_subsequent = 0

        if(base_hit_object.type == "slider"):#TODO check this
            slider_follow_circle_radius = circle_radius * 3
            segment_length = base_hit_object.length // base_hit_object.segment_count
            segment_end_time = base_hit_object.start_time + segment_length

            cursor_pos = base_hit_object.position

            for time in range(base_hit_object.start_time + LAZY_SLIDER_STEP_LENGTH, segment_end_time, LAZY_SLIDER_STEP_LENGTH):
                difference = base_hit_object.position_at_time(time) - cursor_pos
                distance = vector_length(difference)

                if(distance > slider_follow_circle_radius):
                    normalize_vector(difference)
                    distance -= slider_follow_circle_radius
                    cursor_pos += difference * distance
                    self.lazy_slider_length_first += distance
                
            self.lazy_slider_length_first += scaling_factor

            if(base_hit_object.segment_count % 2 == 1):
                self.normalized_end_position = cursor_pos * scaling_factor

            if(base_hit_object.segment_count > 1):
                segment_end_time += segment_length

                for time in range(segment_end_time - segment_length + LAZY_SLIDER_STEP_LENGTH, segment_end_time, LAZY_SLIDER_STEP_LENGTH):
                    difference = base_hit_object.position_at_time(time) - cursor_pos
                    distance = vector_length(difference)

                    if(distance > slider_follow_circle_radius):
                        normalize_vector(difference)
                        distance -= slider_follow_circle_radius
                        cursor_pos += difference * distance
                        self.lazy_slider_length_subsequent += distance
                
                self.lazy_slider_length_subsequent += scaling_factor

                if(base_hit_object.segment_count % 2 == 1):
                    self.normalized_end_position = cursor_pos * scaling_factor

        else:
            self.normalized_end_position = self.base_hit_object.end_position * scaling_factor
    
    def calculate_strains(self, previous_hit_object):
        self.calculate_specific_strain(previous_hit_object, 0)
        self.calculate_specific_strain(previous_hit_object, 1)

    

    def calculate_specific_strain(self, previous_hit_object, type):
        def spacing_weight(distance, type):
            ALMOST_DIAMETER = 90
            STREAM_SPACING_TRESHOLD = 110
            SINGLE_SPACING_TRESHOLD = 125

            if(type == 0):
                weight = 0
                if(distance > SINGLE_SPACING_TRESHOLD): weight = 2.5
                elif(distance > STREAM_SPACING_TRESHOLD): weight = 1.6 + 0.9 * (distance - STREAM_SPACING_TRESHOLD) / (SINGLE_SPACING_TRESHOLD - STREAM_SPACING_TRESHOLD)
                elif(distance > ALMOST_DIAMETER): weight = 1.2 + 0.4 * (distance - ALMOST_DIAMETER) / (STREAM_SPACING_TRESHOLD - ALMOST_DIAMETER)
                elif(distance > (ALMOST_DIAMETER / 2)): weight = 0.95 + 0.25 * (distance - (ALMOST_DIAMETER / 2)) / (ALMOST_DIAMETER / 2)
                else: weight = 0.95

                return weight

            elif(type == 1):
                return pow(distance, 0.99)

            return 0

        SPACING_WEIGHT_SCALING = [1400, 26.25]

        addition = 0
        time_elapsed = self.base_hit_object.start_time - previous_hit_object.base_hit_object.start_time
        decay = pow(DECAY_BASE[type], time_elapsed / 1000)

        if(self.base_hit_object.type == "spinner"):
            pass

        elif(self.base_hit_object.type == "slider"):
            if(type == 0): # speed
                addition = spacing_weight(previous_hit_object.lazy_slider_length_first + previous_hit_object.lazy_slider_length_subsequent * (previous_hit_object.base_hit_object.segment_count - 1) + self.distance_to(previous_hit_object), type) * SPACING_WEIGHT_SCALING[type]
            elif(type == 1):
                addition = (spacing_weight(previous_hit_object.lazy_slider_length_first, type) + spacing_weight(previous_hit_object.lazy_slider_length_subsequent, type) * (previous_hit_object.base_hit_object.segment_count - 1) + spacing_weight(self.distance_to(previous_hit_object), type)) * SPACING_WEIGHT_SCALING[type]

        elif(self.base_hit_object.type == "normal"):
            addition = spacing_weight(self.distance_to(previous_hit_object), type) * SPACING_WEIGHT_SCALING[type]
        
        addition /= max(time_elapsed, 50)
        self.strains[type] = previous_hit_object.strains[type] * decay + addition

    def distance_to(self, other):
        return vector_length((self.normalized_start_position - other.normalized_end_position))

if __name__ == '__main__':
    # args = parse_args()
    for i in range (6) :
        map_path = '../osumapgenerator/data/formatted_beatmaps/maps/{}.osu'.format(i)
        diff, orig = anaylize_difficulty(map_path)
        print("For", i,  " Orig:", orig, "Estimation:", diff)