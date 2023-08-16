import os

import cv2

from src.reprojection import visualize_measurement
from src.track_tool_tips import ToolTipFinder
from src.utils import list_frames


# Method to show an image / zoom in and out / select two pixels / return selected pixel coordinates
def show_image_with_zoom(image, config):
        zoom = 1
        left_clicks_coord = []
        x, y = -1, -1
        h, w = image.shape[:2]
        zoomed = image.copy()
        zoom_window_size = (40, 40)
        zoom_scale = 20
        
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

        # Return the selected points
        return left_clicks_coord[0], left_clicks_coord[1]


# Create user interface to measure distances in online/offline mode
def create_user_interface(config):
    
    if config['user_interface']['method']=='online':
        ttf = ToolTipFinder(config)

    # Get list of paths to frames and frame names
    frame_paths, frame_names = list_frames(path=config['user_interface']['path_left_images'])

    # Path to disparity maps
    disparity_path = config['user_interface']['path_disparity_maps']

    current_frame_index = 0

    while True:
        # Read image
        img = cv2.imread(frame_paths[current_frame_index], 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Frame ' + frame_paths[current_frame_index], (15,15), font, 0.5, (0, 255, 0), 1)
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
            current_frame_index = min(current_frame_index + 1, len(frame_paths) - 1)
        # Measurement trigger
        elif key == ord(" "):
            if config['user_interface']['method'] == 'offline':
                print('Measurment triggered!')
                p1, p2 = show_image_with_zoom(img, config)
                
            elif config['user_interface']['method'] == 'online':
                points, result = ttf.find_tips(image=img)
                
                # Show results in image
                if result:
                    # View points in image
                    print('p1:', points[0])
                    print('p2:', points[1])
                    cv2.circle(img, (points[0][0],points[0][1]), radius=0, color=(255, 0, 0), thickness=8)
                    cv2.circle(img, (points[1][0],points[1][1]), radius=0, color=(255, 0, 0), thickness=8)
                    p1 = points[0]
                    p2 = points[1]
                else:
                    print('Online point selection failed!')
                
                cv2.imshow("Result", img)
                key = cv2.waitKey(0)
            
            # Measure distances between points p1 and p2
            if config['user_interface']['method'] == 'offline' or result:
                visualize_measurement(image_path=frame_paths[current_frame_index],
                                      disparity_map_path=os.path.join(disparity_path,
                                                                      frame_names[current_frame_index][:-4] + '.npy'),
                                      config=config,
                                      point1=p1,
                                      point2=p2)