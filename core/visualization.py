import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import numpy as np
import subprocess as sp


def get_resolution(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]

    try:
        pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
        for line in pipe.stdout:
            w, h = line.decode().strip().split(',')
    finally:
        pipe.stdout.close()

    return int(w), int(h)


def read_video(filename, skip=0, limit=-1):
    w, h = get_resolution(filename)

    command = ['ffmpeg',
               '-i', filename,
               '-f', 'image2pipe',
               '-pix_fmt', 'rgb24',
               '-vsync', '0',
               '-vcodec', 'rawvideo', '-']

    i = 0
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
    try:
        while True:
            data = pipe.stdout.read(w * h * 3)
            if not data:
                break
            i += 1
            if i > skip:
                yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))
            if i == limit:
                break
    finally:
        pipe.stdout.close()


def downsample_tensor(X, factor):
    length = X.shape[0] // factor * factor
    return np.mean(X[:length].reshape(-1, factor, *X.shape[1:]), axis=1)


def render_animation():
    pass
