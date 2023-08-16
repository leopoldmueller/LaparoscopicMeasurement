import errno
import os

import yaml


def is_path_file(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), string)

def load_yaml_data(path):
    if is_path_file(path):
        with open(path) as f_tmp:
            return yaml.load(f_tmp, Loader=yaml.FullLoader)


def list_frames(path, image_format='.png'):
    # Prepare in put stream
    frame_names = os.listdir(path)
    frame_names = [frame for frame in frame_names if frame.endswith(image_format)]
    # Sort frames by their frame index
    frame_names.sort(key=lambda x: int(x.split('.')[0]))
    frame_paths = [os.path.join(path, frame) for frame in frame_names]
    
    # Return list of frame names
    return frame_paths, frame_names