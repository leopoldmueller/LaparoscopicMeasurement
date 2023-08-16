import json
import os
import shutil
import time

import cv2
import pandas as pd

from src.reprojection import Ruler
from src.track_tool_tips import ToolTipFinder
from src.utils import list_frames


# Super class for creating labels (points to be measured)
class DistanceLabler:
    
     # Initilize with config file
    def __init__(self, config):
        
        # Set evaluation mode
        if config['evaluation']['method'] == 'online':
            self.online = True
            
        else:
            self.online = False
        
        # Set path to data directory
        self.path_data = config['evaluation']['path_to_data']
        # Path to disparity maps
        self.path_disparity_maps = os.path.join(self.path_data, 'disparity')
        # Path to left images
        self.path_left = os.path.join(self.path_data, 'left')
        # Path to right images
        self.path_right = os.path.join(self.path_data, 'right')
        # Path to label json
        self.path_json = os.path.join(self.path_data, 'labels.json')
        # Path to labels as images
        self.path_label_images = os.path.join(self.path_data, 'label_images')
        # Load json or create if json does not exist
        self.load_or_create_json()
        # Store labels as image
        self.store_labels_as_image = config['evaluation']['store_labels_as_image']
        
        
    # Load or create json 
    def load_or_create_json(self):
        # Load json
        if not os.path.exists(self.path_json):
            labels = {}
        else:
            # Read JSON file as a dictionary
            with open(self.path_json, 'r') as f:
                labels = json.load(f)
        
        # Store in class variable
        self.labels = labels
    
    
    # Save json 
    def save_json(self):
        # Save the updated dictionary as a JSON file
        with open(self.path_json, 'w') as f:
            json.dump(self.labels, f)
            

# Selects points with yolov8 (online method)
class OnlineDistanceLabler(DistanceLabler):
    
    # Init properties from super class
    def __init__(self, config):
        super().__init__(config)
        self.ttf = ToolTipFinder(config)
        
    def create_labels_for_image(self):
        frame_paths, frame_names  = list_frames(self.path_left)
        
        for i, frame_path in enumerate(frame_paths):
            img = cv2.imread(frame_path, 1)
            points, result = self.ttf.find_tips(image=img)
            
            
            if result:
                self.labels[frame_names[i]] = {'p1':    points[0],
                                               'p2':    points[1]}
            
                if self.store_labels_as_image:  
                    if not os.path.exists(self.path_label_images):
                        os.makedirs(self.path_label_images)              
                    cv2.circle(img, (points[0][0],points[0][1]), radius=0, color=(0, 255, 0), thickness=8)
                    cv2.circle(img, (points[1][0],points[1][1]), radius=0, color=(0, 255, 0), thickness=8)
                    cv2.imwrite(os.path.join(self.path_label_images, frame_names[i]), img)
            else:
                self.labels[frame_names[i]] = 'failed'
            
            self.save_json()
            

# Select points manually (offline method)
class OfflineDistanceLabler(DistanceLabler):
    
    # Init properties from super class
    def __init__(self, config):
        super().__init__(config)
    
    # Method to show an image / zoom in and out / select two pixels / return selected pixel coordinates
    def show_image_with_zoom(self, image):
        zoom = 1
        left_clicks_coord = []
        x, y = -1, -1
        h, w = image.shape[:2]
        zoomed = image.copy()
        zoom_window_size = (40, 40)
        zoom_scale = 20
        
        # Get mouse input
        def on_mouse(event, x_new, y_new, flags, param):
            
            nonlocal zoom, left_clicks_coord, x, y, zoomed
            
            if event == cv2.EVENT_LBUTTONDOWN:
                # Append the coordinates of the selected pixel in the original image to a list
                if zoom == 1:
                    c1 = x_new*zoom
                    c2 = y_new*zoom
                    left_clicks_coord.append((c1, c2))
                if zoom == 2:
                    c1 = x + x_new // zoom_scale - zoom_window_size[1]//2
                    c2 = y + y_new // zoom_scale -zoom_window_size[1]//2 
                    left_clicks_coord.append((c1, c2))
                
                # Show coordinates in image
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, '(' + str(c1) + ',' + str(c2) + ')', (c1,c2), font, 0.5, (255, 255, 0), 1)
                cv2.circle(image, (c1,c2), radius=0, color=(0, 0, 255), thickness=-1)
            
            elif event == cv2.EVENT_RBUTTONDOWN:
                x, y = x_new, y_new
                if zoom == 1:
                    zoom = 2
                    # Get the region around the clicked point
                    x1, y1 = max(0, x - zoom_window_size[0]//2), max(0, y - zoom_window_size[1]//2)
                    x2, y2 = min(w, x + zoom_window_size[0]//2), min(h, y + zoom_window_size[1]//2)
                    zoomed = cv2.resize(image[y1:y2, x1:x2].copy(), None, fx=zoom_scale, fy=zoom_scale, interpolation= cv2.INTER_NEAREST)
                else:
                    zoom = 1
                    zoomed = image.copy()
        
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", on_mouse)
        
        while True:
            cv2.imshow("image", zoomed)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
        cv2.destroyAllWindows()

        if len(left_clicks_coord)==2:
            # Return the selected points
            return left_clicks_coord[0], left_clicks_coord[1]
        else:
            return None, None

    # Create labels for images in current directory  
    def create_labels_for_image(self):
        
        # List frames in current data directory
        frame_paths, frame_names = list_frames(self.path_left)
        
        # Iterate over images and select points
        for i, frame_path in enumerate(frame_paths):
            if not frame_names[i] in self.labels:
                p1, p2 = self.show_image_with_zoom(cv2.imread(frame_path, 1))
                if p1 is None:
                    break
                self.labels[frame_names[i]] = {'p1':    p1,
                                            'p2':    p2}
                self.save_json()

# Iterate over all images and calculate the distances between the labeld points
class MeasurementRunner:
    
    def __init__(self, config):
        self.config = config
        self.result_type = config['evaluation']['measure']['result_type']
        self.path_data_to_measure = config['evaluation']['measure']['path_data_to_measure']
        self.path_left = os.path.join(self.path_data_to_measure, 'left')
        self.path_color = os.path.join(self.path_data_to_measure, 'left')
        self.path_json = os.path.join(self.path_data_to_measure, 'labels.json')
        self.path_disparity = os.path.join(self.path_data_to_measure, 'disparity')
        self.measurement_id = int(time.time())
        # Load json and store it as self.labels
        self.load_json()
        # Create experiment directory
        self.initialize_experiment()
        
    # Create experiment directory with metadata, results_csv
    def initialize_experiment(self):
        
        self.hyperparameters = {'measurement_id': self.measurement_id,
                                'method': self.config['evaluation']['method'],
                                'every_nth': self.config['evaluation']['measure']['every_nth'],
                                'disparity_map_type': self.config['evaluation']['measure']['disparity_map_type'],
                                'surface_reconstruction': self.config['evaluation']['measure']['surface_reconstruction'],
                                'spline_s': self.config['evaluation']['measure']['spline_s'],
                                'offset': self.config['evaluation']['measure']['offset']
                                }
        
        columns = ['measurement_id',
                   'method',
                   'disparity_map_type',
                   'surface_reconstruction',
                   'offset',
                   'every_nth',
                   'spline_s',
                   'frame_name',
                   'ground_truth',
                   'euclidian',
                   'on_surface',
                   'on_surface_spline']
        
        # Individual csv for each dataset
        if self.result_type == 'individual':
            
            self.path_experiment = os.path.join(self.path_data_to_measure, f"experiment_{self.measurement_id}")
            if not os.path.exists(self.path_experiment):
                os.makedirs(self.path_experiment)
            
            # Save the config of the experiment configuration as a JSON file
            with open(os.path.join(self.path_experiment, 'configuration.json'), 'w') as f:
                json.dump(self.config['evaluation'], f, indent=4)
                
            self.df_results = pd.DataFrame(columns=columns)
            self.ground_truth = self.config['evaluation']['measure']['ground_truth']
        
        # All results are stored in on csv
        elif self.result_type == 'all_in_one':
            if not os.path.exists(os.path.join('datasets','evaluation','results.csv')):
                self.df_results = pd.DataFrame(columns=columns)
            else:
                self.df_results = pd.read_csv(os.path.join('datasets','evaluation','results.csv'))
    # Load json
    def load_json(self):
        # Load json
        with open(self.path_json, 'r') as f:
            labels = json.load(f)
        
        # Store in class variable
        self.labels = labels

    # Store results of actual measurement in dataframe
    def store_results(self, measurement_results):
        values = {**measurement_results, **self.hyperparameters}
        self.df_results.loc[len(self.df_results)] = values
    
    # Measure distance for given points and frame
    def measure_distance(self, p1, p2, frame_name):
        
        ruler = Ruler(point1=p1, point2=p2, config=self.config, frame_name=frame_name, image_path=os.path.join(self.path_left, frame_name), disparity_map_path=os.path.join(self.path_disparity, frame_name[:-3] + 'npy'))
        measurement_result = ruler.measure()
        self.store_results(measurement_result)
        
    # Process all frames
    def process_frames(self):
        i = 0
        # Iterate over all frames and their selected points
        for frame_name in self.labels.keys():
            i += 1
            print(f"Process image {i} of {len(self.labels)}")
            if self.labels[frame_name] != 'failed':
                self.measure_distance(p1=self.labels[frame_name]['p1'], p2=self.labels[frame_name]['p2'], frame_name=frame_name)
        
        if self.result_type == 'individual':
            self.df_results.to_csv(os.path.join(self.path_experiment, 'results.csv'), index=False)
        
        elif self.result_type == 'all_in_one':
            self.df_results.to_csv(os.path.join(os.path.join('datasets', 'evaluation', 'results.csv')), index=False)

# Iterate over dataset and select images for evaluation   
class ImageSelector():
    
    def __init__(self, config):
        
        # Set path to source data directory
        self.path_source_data = config['evaluation']['path_source_data']
        # Path to source disparity maps
        self.path_source_disparity_maps = os.path.join(self.path_source_data, 'disparity')
        # Path to source left images
        self.path_source_left = os.path.join(self.path_source_data, 'left')
        # Path to source right images
        self.path_source_right = os.path.join(self.path_source_data, 'right')
        
        # Set path to  target data directory
        self.path_target_data = config['evaluation']['path_target_data']
        # Path to disparity maps
        self.path_target_disparity_maps = os.path.join(self.path_target_data, 'disparity')
        # Path to left images
        self.path_target_left = os.path.join(self.path_target_data, 'left')
        # Path to right images
        self.path_target_right = os.path.join(self.path_target_data, 'right')
        
        # Create target dircetories
        self.create_target_dirs()
        
    
    # Method to create target directories if the don't exist
    def create_target_dirs(self):
        
        # Create target directories if they don't exist
        if not os.path.exists(self.path_target_disparity_maps):
            os.makedirs(self.path_target_disparity_maps)
        if not os.path.exists(self.path_target_left):
            os.makedirs(self.path_target_left)
        if not os.path.exists(self.path_target_right):
            os.makedirs(self.path_target_right)
    
    def run_through_directory(self):
        # Get list of paths to frames and frame names
        source_frame_paths, source_frame_names = list_frames(path=self.path_source_left)

        current_frame_index = 0

        while True:
            # Read image
            img = cv2.imread(source_frame_paths[current_frame_index], 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, 'Frame ' + source_frame_paths[current_frame_index], (15,15), font, 0.5, (0, 255, 0), 1)
            # Show image in window
            cv2.imshow('frame', img)
            
            key = cv2.waitKey(1) & 0xFF

            # Exit tool
            if key == ord("q"):
                break
            # Previous frame
            elif key == ord("a"):
                current_frame_index = max(current_frame_index - 1, 0)
            # Next frame
            elif key == ord("d"):
                current_frame_index = min(current_frame_index + 1, len(source_frame_paths) - 1)
            # With pressing "blank" the actual frame is selected and added to the target directory
            elif key == ord(" "):
                shutil.copy2(os.path.join(self.path_source_left, source_frame_names[current_frame_index]), os.path.join(self.path_target_left, source_frame_names[current_frame_index]))
                shutil.copy2(os.path.join(self.path_source_right, source_frame_names[current_frame_index]), os.path.join(self.path_target_right, source_frame_names[current_frame_index]))
                shutil.copy2(os.path.join(self.path_source_disparity_maps, source_frame_names[current_frame_index][:-3] + 'npy'), os.path.join(self.path_target_disparity_maps, source_frame_names[current_frame_index][:-3] + 'npy'))
                shutil.copy2(os.path.join(self.path_source_disparity_maps, source_frame_names[current_frame_index]), os.path.join(self.path_target_disparity_maps, source_frame_names[current_frame_index]))
                print(f'{len(os.listdir(self.path_target_left))} images in directory.')
    

# Run evaluation tasks
def run_evaluation(config):
    
    # Select images
    if config['evaluation']['task']=='select_images':
        # Videos to select from
        video_dirs = ['allinone', 'bowel', 'bowel_2', 'bowel_3', 'bowel_4', 'colorized1', 'colorized2', 'colorized3', 'colorized4', 'colorized5', 'on_surface4', 'phantom', 'on_surface3', 'hhl', 'hhl2', 'hhs', 'on_surface', 'on_surface2', 'on_surface5', 'on_surface6']
        for dir in video_dirs:
            print(dir)
            config['evaluation']['path_source_data'] = os.path.join(config['evaluation']['path_source_root'], dir, 'infrared')
            ims = ImageSelector(config)
            ims.run_through_directory()
    
    # Create labels with yolov8
    elif config['evaluation']['task']=='create_labels' and config['evaluation']['method']=='online':
        odl = OnlineDistanceLabler(config)
        odl.create_labels_for_image()
    
    # Create labels manually
    elif config['evaluation']['task']=='create_labels' and config['evaluation']['method']=='offline':
        odl = OfflineDistanceLabler(config)
        odl.create_labels_for_image()
    
    # Perform grid search over search space
    elif config['evaluation']['task']=='measure' and config['evaluation']['measure']['result_type']=='all_in_one':
        # Grid search over search space
        for disparity_map_type in config['evaluation']['search_space']['disparity_map_type']:
            config['evaluation']['measure']['disparity_map_type'] = disparity_map_type
            for surface_reconstruction in config['evaluation']['search_space']['surface_reconstruction']:
                config['evaluation']['measure']['surface_reconstruction'] = surface_reconstruction
                for offset in config['evaluation']['search_space']['offset']:
                    config['evaluation']['measure']['offset'] = offset
                    for every_nth in config['evaluation']['search_space']['every_nth']:
                        config['evaluation']['measure']['every_nth'] = every_nth
                        for spline_s in config['evaluation']['search_space']['spline_s']:
                            config['evaluation']['measure']['spline_s'] = spline_s
                            # Run measurement
                            mr = MeasurementRunner(config=config)
                            mr.process_frames()
    
    # Process labels and images and store results
    elif config['evaluation']['task']=='measure':
        mr = MeasurementRunner(config=config)
        mr.process_frames()
    else:
        print('Wrong task. Choose between select_images, create_labels and calculate_distances')