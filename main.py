'''This is the main.py of surg_dimes. By running this file, you can use the dimension measurement framework.'''
from src.create_disparity_maps import run_RAFT_stereo
from src.evaluate_images import run_evaluation
from src.track_tool_tips import test_model
from src.user_interface import create_user_interface
from src.utils import load_yaml_data

config = load_yaml_data("config.yaml")

# Set run raftstereo to True to create disparity maps for actual configuration
if config['raftstereo']['run']:
    run_RAFT_stereo(config)

if config['user_interface']['run']:
    create_user_interface(config)

if config['evaluation']['run']:
    run_evaluation(config=config)

if config['yolov8']['test']:
    test_model(config=config)