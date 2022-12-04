import random
import os
import librosa

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
ENCODING = "utf-8"


def random_generator(save_path, file_name, end_time, gap_limitation, start_limitation):
    """
    save_path: thr root for saving the generated files
    file_name: the name of the file that we want to write in the generated info
    end_time: the length of the audio file (millisecond)
    gap_limitation: [0, 1000] the interval of the gap between two beats
    start_limitation: [0, 10000] the interval of the first beat
    """
    with open(save_path + file_name, "w+", encoding=ENCODING) as f:
        f.write("[HitObjects]"+ "\n")
        cur_time = random.randint(0, start_limitation)

        while cur_time < end_time:
            new_line = str(random.randint(0, WINDOW_WIDTH)) + "," + str(random.randint(0, WINDOW_HEIGHT)) + "," + str(cur_time) + "\n"
            f.write(new_line)
            cur_time += random.randint(0, gap_limitation)


def meta_data(read_path, save_path, file_name):
    """
    read_path: the root of the human generated ground truth osu files with the meta data
    save_path: the root of the randomly generated osu files
    file_name: the name of the osu file 
    """
    hitobjects_info = open(save_path + file_name, "r+", encoding=ENCODING).read()
    meta_data = open(read_path + file_name, "r+", encoding=ENCODING).read().split("[HitObjects]\n")[0]
    with open(save_path + file_name, "w+", encoding=ENCODING) as f:
        f.write(meta_data + hitobjects_info)


def song_map(mapping):
    """
    mapping: the mapping file with the relation between song files and map files
    """
    with open(mapping, "r+", encoding=ENCODING) as f:
        map_dict = {}
        maps = f.readlines()[1:]

        for i in maps:
            value, key = i[:-1].split(",")
            map_dict[key] = value
    return map_dict


if __name__ == '__main__':
    cur_path = os.getcwd().replace("\\", "/")
    song_path = cur_path + "/evaluation/songs/"
    song_names = os.listdir(song_path)

    map_dict = song_map(cur_path + "/evaluation/song_mapping.csv")

    for song_name in song_names:

        duration = librosa.get_duration(filename=song_path+song_name)
        # random_generator(cur_path + "/random/", map_dict[song_name], duration * 1000, 1000, 10000)
        meta_data(cur_path + "/evaluation/maps/", cur_path + "/evaluation/random/", map_dict[song_name])