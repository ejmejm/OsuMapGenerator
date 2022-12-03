import os
import numpy as np
import lzma

NUM_ATTRIBUTE_HITOBJECT = 3
ENCODING = "utf-8"


def hitobjects_info(info):
    """
    info: a string that contain all the information in the file
    """
    lines = info.split("\n")[:-1]
    flag = False
    info = []
    for line in lines:
        if flag:
            info.append(",".join(line.split(",")[:NUM_ATTRIBUTE_HITOBJECT]))
        if line == "[HitObjects]":
            flag = True
    return info


def time_diff(gen_info, gt_info):
    """
    gen_info: a string that contain the information in the generated osu file
    gt_info: a string that contain the information in the ground truth osu file
    """
    gen_h_info = hitobjects_info(gen_info)
    gt_h_info = hitobjects_info(gt_info)

    gen = [int(line.split(",")[2]) for line in gen_h_info]
    gt = [int(line.split(",")[2]) for line in gt_h_info]

    count = 0

    for time in gt:
        count += np.min(np.abs(np.array(gen) - time))

    return count / len(gt)


def compression_dist(gen_info, gt_info):
    """
    gen_info: a string that contain the information in the generated osu file
    gt_info: a string that contain the information in the ground truth osu file
    """
    gen_h_info = hitobjects_info(gen_info)
    gt_h_info = hitobjects_info(gt_info)

    gen_info_str = "\n".join(gen_h_info)
    gt_info_str = "\n".join(gt_h_info)

    gen_byte = gen_info_str.encode(ENCODING)
    gt_byte = gt_info_str.encode(ENCODING)
    gen_gt = gen_byte + gt_byte  # the concatenation of files

    gen_comp = lzma.compress(gen_byte)  # compress file 1
    gt_comp = lzma.compress(gt_byte)  # compress file 2
    gen_gt_comp = lzma.compress(gen_gt)  # compress file concatenated

    cd = (len(gen_gt_comp) - min(len(gen_comp), len(gt_comp))) / max(len(gen_comp), len(gt_comp))

    return cd


if __name__ == '__main__':
    cur_root = os.getcwd().replace("\\", "/")
    gen_root = cur_root + "/random" #"/raw_maps"
    gt_root = cur_root + "/maps"

    for file_name in os.listdir(gen_root):
        gen_file_n = gen_root + "/" + file_name
        gt_file_n = gt_root + "/" + file_name

        gen_info = open(gen_file_n, "r+", encoding=ENCODING).read()
        gt_info = open(gt_file_n, "r+", encoding=ENCODING).read()
        t_diff = time_diff(gen_info, gt_info)
        cd = compression_dist(gen_info, gt_info)