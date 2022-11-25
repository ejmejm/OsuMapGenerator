from pydub import AudioSegment
import os
import numpy as np
import sys
import os
import random
from essentia.standard import MonoLoader, Windowing, Spectrum, MelBands
import torch

WINDOW = None
SPECTRUM = None
MEL = None
FRAME_SIZE = None
TARGET_FRAME_SIZE = 1536

def process_song(song_name, config):
    '''
    process the given song, analyze it, then create a chart for it
    Usage: audio_processing.py song_file.mp3
    '''

    dir_name = song_name + " segments"

    song = song_name[:len(song_name) - 4]
    npy_file = song + " Input.npy"
    dir_list = os.listdir()

    if dir_name not in dir_list:
        print("Segmenting song..")
        segment_song(song_name, config['segment_length'])
    else:
        print("Song already segmented, moving on..")

    print("Processing song..")

    return filter_song(npy_file, dir_name)

def load_audio(audio_path, start_time=None, duration=None):
    """Load audio from file, time in seconds."""
    return AudioSegment.from_file(
        audio_path, 'mp3', start_second=start_time, duration=duration)

def segment_song(song_name, seg_length):
    no_prefix = song_name[:len(song_name) - 4]
    song = AudioSegment.from_mp3(song_name)
    directory_name = song_name + " segments"

    os.mkdir(directory_name)

    i = 1
    current_segment = song[:seg_length]
    current_segment.export(directory_name + "/" + no_prefix + ' ' + str(i) + ".wav", format = "wav", bitrate = "192k")
    i += 1
    
    while len(current_segment) == seg_length:
        current_segment = song[seg_length * i : seg_length * (i + 1)]
        current_segment.export(directory_name + "/" + no_prefix + ' ' + str(i) + ".wav", format = "wav", bitrate = "192k")
        i += 1

    current_segment = np.zeros(seg_length, dtype=np.float64)
    current_segment[:len(song) - (seg_length * i)] = song[len(song) - (seg_length * i):]
    current_segment.export(directory_name + "/" + no_prefix + ' ' + str(i) + ".wav", format = "wav", bitrate = "192k")
    print("Success! number of segments:", str(i))

def create_analyzers(fs=44100.0,
                     nffts=[1024, 2048, 4096],
                     mel_nband=80,
                     mel_freqlo=27.5,
                     mel_freqhi=16000.0,
                     input_size=None,
                     zero_padding=0):
    window = Windowing(
        size=nffts[0], type='blackmanharris62', zeroPadding=zero_padding)
    spectrum = Spectrum(size=nffts[0])
    mel = MelBands(inputSize=input_size or (nffts[0] // 2) + 1,
                    numberBands=mel_nband,
                    lowFrequencyBound=mel_freqlo,
                    highFrequencyBound=mel_freqhi,
                    sampleRate=fs)

    return window, spectrum, mel

def filter_song(file_name = None, dir_name = None):
    file_list = os.listdir()
    for f in file_list:
        if f == file_name:
            print("Song was already processed, exiting filtering..")
            return np.load(file_name, mmap_mode="r")

    cwd = os.getcwd()
    if dir_name != None:
        new_dir = cwd + "/" + dir_name
        os.chdir(new_dir)

    file_list = os.listdir()
    window, spectrum, mel = create_analyzers()
    feats_list = []
    i = 0
    
    for fn in file_list:
        if fn[len(fn) - 1] != 'v':
            continue
        try:
            loader = MonoLoader(filename=fn, sampleRate=44100.0)
            samples = loader()
            feats = window(samples)
            if len(feats) % 2 != 0:
                feats = np.delete(feats, random.randint(0, len(feats) - 1))
            feats = spectrum(feats)
            feats = mel(feats)
            feats_list.append(feats)
            i+=1
        except Exception as e:
            feats_list.append(np.zeros(80, dtype=np.float32))
            i += 1

    # Apply numerically-stable log-scaling
    feats_list = np.array(feats_list)
    feats_list = np.log(feats_list + 1e-16)
    print(len(feats_list), "length of feats list")
    print(type(feats_list[0][0]))
    if dir_name != None:
        os.chdir(cwd)
    np.save(file_name, feats_list)
    return feats_list

def audio_to_np(segment, n_splits=1, n_mel_bands=80, sample_rate=44100):
    global WINDOW, SPECTRUM, MEL, FRAME_SIZE
    
    # Pad to make all segments the same length
    padding_size = int(np.ceil(len(segment) / n_splits) * n_splits - len(segment))
    segment = np.pad(segment, (0, padding_size), 'constant')
    feats_list = np.split(segment.astype(np.float32), n_splits)
    # All windows must have an even length, truncate where needed
    feats_list = [f if len(f) % 2 == 0 else f[:-1] for f in feats_list]

    frame_size = len(feats_list[0])

    # Normalize all segments to be the same size with 0 padding except
    # for the special case where they are greater than the target size,
    # then recreate all the analyzers
    if WINDOW is None:
        if frame_size > TARGET_FRAME_SIZE:
            WINDOW, SPECTRUM, MEL = create_analyzers(
                nffts=[frame_size], fs=sample_rate, mel_nband=n_mel_bands)
        else:
            WINDOW, SPECTRUM, MEL = create_analyzers(
                nffts=[TARGET_FRAME_SIZE], fs=sample_rate, mel_nband=n_mel_bands,
                zero_padding=TARGET_FRAME_SIZE-frame_size)
    elif frame_size != FRAME_SIZE:
        if frame_size > TARGET_FRAME_SIZE:
            WINDOW, SPECTRUM, MEL = create_analyzers(
                nffts=[frame_size], fs=sample_rate, mel_nband=n_mel_bands)
        elif FRAME_SIZE > TARGET_FRAME_SIZE:
            WINDOW, SPECTRUM, MEL = create_analyzers(
                nffts=[TARGET_FRAME_SIZE], fs=sample_rate, mel_nband=n_mel_bands,
                zero_padding=TARGET_FRAME_SIZE-frame_size)
        else:
            WINDOW = Windowing(
                size=TARGET_FRAME_SIZE, type='blackmanharris62',
                zeroPadding=TARGET_FRAME_SIZE-frame_size)
    FRAME_SIZE = frame_size

    for i in range(len(feats_list)):
        feats_list[i] = WINDOW(feats_list[i])
        feats_list[i] = SPECTRUM(feats_list[i])
        feats_list[i] = MEL(feats_list[i])

    feats_arr = np.stack(feats_list)
    
    # Apply numerically-stable log-scaling
    feats_arr = np.log(feats_arr + 1e-16)
    return feats_arr

def prepare_audio_tensor(audio, config, device=None):
  audio = [np.pad(
    a, ((0, config['max_hit_objects'] - len(a)), (0, 0), (0, 0))) \
    for a in audio]
  audio = torch.from_numpy(np.stack(audio)).float()
  audio = audio.to(device or config['device'])
  audio = audio.transpose(2, 3) # (batch, n_tokens, channel, audio_len)
  return audio