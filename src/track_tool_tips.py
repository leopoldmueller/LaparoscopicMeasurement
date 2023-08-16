import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image
from scipy.spatial.distance import cdist
from ultralytics import YOLO


# Class for automatic measurement
class ToolTipFinder:
    
    # Initilize with config file
    def __init__(self, config):
        
        self.model = YOLO(config['yolov8']['weights'])
        self.save = config['yolov8']['save']
        self.result = True
        
    def calculate_tips(self, mask):
        # Compute the coordinates of the non-zero pixels in the mask
        y_coords, x_coords = np.where(mask != 0)

        # Compute the centroid of the mask
        x_center = int(np.mean(x_coords))
        y_center = int(np.mean(y_coords))
        
        # Compute the distances from each pixel to the center
        distances = cdist(np.column_stack((x_coords, y_coords)), [(x_center, y_center)])

        # Find the pixel with the largest distance to the center
        max_idx = np.argmax(distances)

        # Get the coordinates of the pixel with the largest distance
        x_max, y_max = x_coords[max_idx], y_coords[max_idx]#
        
        # Rescale coordinates to original values
        x_max = int(round(x_max * self.image_size_x / mask.shape[1]))
        y_max = int(round(y_max * self.image_size_y / mask.shape[0]))
        
        return (x_max, y_max)
        
    # Run yolov8 for given image to predict tool segmentations and calculate their tips
    def find_tips(self, image):
        
        # Initialize the tooltip array
        points = []
        
        # Store original image size for rescaling the tooltip coordinates
        self.image_size_x = image.shape[1] 
        self.image_size_y = image.shape[0]
        
        # Get results from yolov8
        results = self.model.predict(source=image, save=self.save, stream=False)  # save plotted images

        # Extract masks from results and store as numpy array
        masks = results[0].masks
        if not masks is None:
            masks = masks.data.cpu().numpy()       
        
            # For each found instrument calculate the tooltip
            for mask in masks:
                points.append(self.calculate_tips(mask))
        
        # Print how many tools (tooltips) we found
        print(f"Found {len(points)} tools.")
        
        # If the number of tools is not as exspected (=2) set self.result to false
        if len(points) != 2:
            self.result = False
            print('Invalid points: ', points)
        
        else:
            self.result = True
        
        # Return the tooltips and information of the result
        return points, self.result


# Class to test the segmentation on image / webcam stream / realsense stream (left infrared input stream)
class ModelTester:
    
    # Initilize with config file
    def __init__(self, config):
        
        self.model = YOLO(config['yolov8']['weights'])
        self.save = config['yolov8']['save']
        self.show_results = config['yolov8']['show']
        self.path = config['yolov8']['image_path']
        self.iou = config['yolov8']['iou']
        self.conf = config['yolov8']['conf']
        self.mode = config['yolov8']['mode']
        
    # Start intel realsense input stream and predict on every frame
    def run_model_on_realsense_stream(self):
                     
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)

        pipeline.start(config)

        try:
            while True:
                frames = pipeline.wait_for_frames()
                infrared_frame = frames.get_infrared_frame(1)
                if not infrared_frame:
                    continue
                # Convert the frame to an OpenCV image
                img = np.asanyarray(infrared_frame.get_data())
                print(img.shape)
                self.model.predict(source=cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), save=False, show=True)
        finally:
            pipeline.stop()
    
    # Run yolov8 on a sample image
    def run_model_on_sample_image(self):
        self.model.predict(source=self.path, save=self.save, show=self.show_results, conf=self.conf, iou=self.iou)
    
    # Run yolov8 on webcam stream
    def run_model_on_webcam(self):
        self.model.predict(source=2, save=False, show=True)
        

# Method for testing a yolov8 model on image / webcam stream / realsense stream (left infrared input stream)
def test_model(config):
    
    mt = ModelTester(config)
    
    if mt.mode == 'webcam':
        mt.run_model_on_webcam()
    elif mt.mode == 'realsense':
        mt.run_model_on_realsense_stream()
    elif mt.mode == 'iamge':
        mt.run_model_on_sample_image()
    
