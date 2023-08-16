import cv2
import matplotlib.cm as plt
import networkx as nx
import numpy as np
import open3d as o3d
from scipy.interpolate import splev, splprep

from src.utils import load_yaml_data


# Contains all camera parameters
class CamPara:
    
    # Initilize with config file
    def __init__(self, config):
        config = load_yaml_data(config['camera']['path_camera_specs'])
        # Intrinsics
        self.fx = config["intrinsics"]["fx"]
        self.fy = config["intrinsics"]["fy"]
        self.b= config["intrinsics"]["b"]
        self.f = config["intrinsics"]["f"]
        self.cx = config["intrinsics"]["cx"]
        self.cy = config["intrinsics"]["cy"]
        self.imx = config["intrinsics"]["imy"]
        self.imy = config["intrinsics"]["imx"]
        
        # Extrinsics
        # Left camera rotations
        self.lrxx = config["extrinsics"]["lrxx"]
        self.lrxy = config["extrinsics"]["lrxy"]
        self.lrxz = config["extrinsics"]["lrxz"]
        self.lryx = config["extrinsics"]["lryx"]
        self.lryy = config["extrinsics"]["lryy"]
        self.lryz = config["extrinsics"]["lryz"]
        self.lrzx = config["extrinsics"]["lrzx"]
        self.lrzy = config["extrinsics"]["lrzy"]
        self.lrzz = config["extrinsics"]["lrzz"]
        
        # Right camera rotations
        self.rrxx = config["extrinsics"]["rrxx"]
        self.rrxy = config["extrinsics"]["rrxy"]
        self.rrxz = config["extrinsics"]["rrxz"]
        self.rryx = config["extrinsics"]["rryx"]
        self.rryy = config["extrinsics"]["rryy"]
        self.rryz = config["extrinsics"]["rryz"]
        self.rrzx = config["extrinsics"]["rrzx"]
        self.rrzy = config["extrinsics"]["rrzy"]
        self.rrzz = config["extrinsics"]["rrzz"]

    # Print all parameters
    def print_par(self):
        print("fx:  ",  self.fx)
        print("fy:  ",  self.fy)
        print("b:  ",  self.b)
        print("f:  ",  self.f)
        print("cx:  ",  self.cx)
        print("cy:  ",  self.cy)

    # Return intrinsics as matrix
    def get_intrinsics(self):

        return np.array([[self.fx, 0, self.cx],
                        [0, self.fy, self.cy],  
                        [0, 0, 1]])
    
    # Get extrinsics as matrix (default is left camera)
    def get_extrinsics(self, camera=0):
        
        if camera == 0: 
            extrinsics = np.array([[self.lrxx, self.lrxy, self.lrxz, self.b],
                                   [self.lryx, self.lryy, self.lryz, 0],
                                   [self.lrzx, self.lrzy, self.lrzz, 0],
                                   [0, 0, 0, 1]])
        
        else:
            extrinsics = np.array([[self.rrxx, self.rrxy, self.rrxz, self.b],
                                   [self.rryx, self.rryy, self.rryz, 0],
                                   [self.rrzx, self.rrzy, self.rrzz, 0],
                                   [0, 0, 0, 1]])

        return extrinsics

    def get_q(self):
        return np.array([[1, 0, 0, -self.cx],
                         [0, 1, 0, -self.cy],
                         [0, 0, 0, self.fx],
                         [0, 0, -1/self.b, 0]])



# Loads disparity map as numpy array from path
def get_disparity_from_path(path_to_disparity_map:str):

    disparity_map = np.load(path_to_disparity_map)

    return disparity_map


# Reproject the pixel coordinates to world coordinates using the open cv built in method.
def reproject_to_world(camera_parameters:object, disparity_map:np.ndarray):
    return cv2.reprojectImageTo3D(disparity_map, camera_parameters.get_q()), cv2.reprojectImageTo3D(disparity_map, camera_parameters.get_q()).reshape(480*848,3)


# Calculate the euclidian distance between two points and return it
def get_euclidian_distance(point1, point2):
    distance = np.linalg.norm(point1 - point2)
    return distance


def compute_shortest_path(mesh, source, target):
    # Convert mesh to a graph
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    graph = nx.Graph()
    graph.add_nodes_from(range(vertices.shape[0]))
    for face in faces:
        graph.add_edge(face[0], face[1], weight=np.linalg.norm(vertices[face[0]] - vertices[face[1]]))
        graph.add_edge(face[1], face[2], weight=np.linalg.norm(vertices[face[1]] - vertices[face[2]]))
        graph.add_edge(face[2], face[0], weight=np.linalg.norm(vertices[face[2]] - vertices[face[0]]))

    # Find the closest vertices to the source and target points
    source_vertex = np.argmin(np.sum((vertices - source)**2, axis=1))
    target_vertex = np.argmin(np.sum((vertices - target)**2, axis=1))

    # Use Dijkstra's algorithm to find the shortest path between the two vertices
    path = nx.dijkstra_path(graph, source_vertex, target_vertex)

    # Compute the length of the shortest path
    length = sum(graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))

    # Return the shortest path and its length
    return path, length


# Compute euclidian, on surface and smoothed on surface distance
def compute_distances(mesh, path, every_nth:int=10, s:float=0.1):

    vertices = mesh.vertices
    path_points = np.array([vertices[i] for i in path])
    # If smooth is set True, do spline interpolation 
    tck, u = splprep(path_points.T, u=None, s=s)
    smoothed_path_points = splev(np.linspace(0, 1, len(path) // every_nth), tck)
    smoothed_path_points = np.array(smoothed_path_points)
    path_points = path_points.transpose()

    print('Euclidian distance in mm: ', np.linalg.norm(path_points.transpose()[0] - path_points.transpose()[-1]))
    print('On-surface distance in mm: ', np.cumsum(np.sqrt(np.sum((path_points.transpose()[1:] - path_points.transpose()[:-1])**2, axis=1)))[-1])
    print('On-surface smoothed distance in mm: ', np.cumsum(np.sqrt(np.sum((smoothed_path_points.transpose()[1:] - smoothed_path_points.transpose()[:-1])**2, axis=1)))[-1])

    return path_points.transpose(), smoothed_path_points.transpose(), np.array([path_points.transpose()[0],path_points.transpose()[-1]])



def create_line_set(points, color):
    # Create a LineSet
    line_set = o3d.geometry.LineSet()
    # Add the points
    line_set.points = o3d.utility.Vector3dVector(points)
    # Create the indix combinations wich point to wich point connects
    lines = np.arange(len(points) - 1)[..., np.newaxis]
    lines = np.concatenate((lines, lines + 1), axis=1)
    # Convert the indices to integer
    lines.astype(int)
    # Add the line connections
    line_set.lines = o3d.utility.Vector2iVector(lines)
    # Add color to line set so we can identify them
    line_set.paint_uniform_color(color)
    # Return the line set
    return line_set


# Method for cropping point cloud section (are between measurement points) to speed up meshing process
def crop_point_cloud_by_selected_points(point1, point2, point_cloud, offset):
    
    # Define bounding box corners
    xmin = min(point1[0], point2[0]) - offset
    xmax = max(point1[0], point2[0]) + offset
    ymin = min(point1[1], point2[1]) - offset
    ymax = max(point1[1], point2[1]) + offset
    zmin = float('-inf')
    zmax = float('inf')

    # Extract points within bounding box
    indices = np.where((point_cloud[:,:,0] > xmin) & (point_cloud[:,:,0] < xmax) &
                    (point_cloud[:,:,1] > ymin) & (point_cloud[:,:,1] < ymax) &
                    (point_cloud[:,:,2] > zmin) & (point_cloud[:,:,2] < zmax))
    
    # Filter point cloud by indices
    filtered_point_cloud = point_cloud[indices]
    
    # Return the filtered (cropped) point cloud and the indices
    return filtered_point_cloud, indices


# Create a o3d marker for a given point
def create_markers_for_points(point1, point2):
    # Create two markers
    marker1 = o3d.geometry.TriangleMesh.create_sphere(radius=1)
    marker2 = o3d.geometry.TriangleMesh.create_sphere(radius=1)

    # Translate markers to specific point in point cloud
    marker1.translate(point1)
    marker2.translate(point2)

    # Assign the color to the vertex_colors property of the marker
    marker1.paint_uniform_color([1, 0, 0])
    marker2.paint_uniform_color([1, 0, 0])

    return marker1, marker2


def get_colors_for_points(image_path, crop, indices=None):
    
    # Read the colors by path
    colors = cv2.imread(image_path, 1)

    if crop:
        colors = colors[indices]
        
    else:
        colors = colors.reshape(-1,3)
    
    return colors

def visualize_measurement(image_path:str, disparity_map_path:str, config, point1, point2):
    
    # Load disparity map from path
    disparity_map = get_disparity_from_path(disparity_map_path)
    
    # Get reprojected pixel coordinates as 3d point cloud
    point_cloud, points = reproject_to_world(camera_parameters=CamPara(config), disparity_map=disparity_map)
    
    # Transform the pixel coordinates into world coordinates (note that the pc is in (y,x) and the pixels are in (x,y) format)
    point1 = point_cloud[point1[1], point1[0]]
    point2 = point_cloud[point2[1], point2[0]]
    	
    # Crop pointcloud by relevant area (area between selected points)
    if config['user_interface']['crop']:
        points, indices = crop_point_cloud_by_selected_points(point1=point1, point2=point2, point_cloud=point_cloud, offset=config['user_interface']['offset'])
    
        colors = get_colors_for_points(image_path=image_path, crop=config['user_interface']['crop'], indices=indices)
        
    else:
        colors = get_colors_for_points(image_path=image_path, crop=config['user_interface']['crop'])
    
    # Create an Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Add colors of image to point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors/255)
    
    marker1, marker2 = create_markers_for_points(point1=point1, point2=point2)

    # Draw point cloud and markers
    if config['user_interface']['show_point_cloud']:
        o3d.visualization.draw_geometries([pcd, marker1, marker2])

    # Now we need a mesh instead of a point cloud to measure the distances on a surface path
    if config['user_interface']['surface_reconstruction'] == 'poisson':
        # Create mesh with poisson
        pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # Invalidate existing normals
        pcd.estimate_normals()  # Estimate normals
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd,
                                                                                    depth=config['user_interface']['poisson']['depth'],
                                                                                    width=config['user_interface']['poisson']['width'],
                                                                                    scale=config['user_interface']['poisson']['scale'],
                                                                                    linear_fit=config['user_interface']['poisson']['linear_fit'])
        mesh.compute_vertex_normals()
        
        if config['user_interface']['poisson']['visualize_mesh']:
            # Visualize the mesh
            o3d.visualization.draw_geometries([mesh, marker1, marker2])

        if config['user_interface']['poisson']['densities_optimization']:
            densities = np.asarray(densities)
            density_colors = plt.get_cmap('plasma')(
                (densities - densities.min()) / (densities.max() - densities.min()))
            density_colors = density_colors[:, :3]
            density_mesh = o3d.geometry.TriangleMesh()
            density_mesh.vertices = mesh.vertices
            density_mesh.triangles = mesh.triangles
            density_mesh.triangle_normals = mesh.triangle_normals
            density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
            o3d.visualization.draw_geometries([density_mesh])

            vertices_to_remove = densities < np.quantile(densities, 0.01)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            print(mesh)
            o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    elif config['user_interface']['surface_reconstruction'] == 'pivoting':
        # Create mesh with pivoting balls
        pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # Invalidate existing normals
        pcd.estimate_normals()  # Estimate normals
        radii = [0.005, 0.01, 0.02, 0.04]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
        mesh.compute_vertex_normals()
        # Visualize the mesh
        o3d.visualization.draw_geometries([mesh])

    elif config['user_interface']['surface_reconstruction'] == 'alpha':
        pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # Invalidate existing normals
        pcd.estimate_normals()
        alpha = 0.03
        print(f"alpha={alpha:.3f}")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    else:
        print('No surface reconstruction choosen!')

    path, length = compute_shortest_path(mesh=mesh, source=point1, target=point2)
    # Convert path into a list of points
    path_points = np.array([mesh.vertices[i] for i in path])

    geometries = [pcd, marker1, marker2]
    
    # Colors for lines
    colors = [[0, 1, 1], [1, 0, 1],[0, 1, 0]]

    for points in [points for points in compute_distances(mesh=mesh, path=path)]:
        geometries.append(create_line_set(points=points, color=colors.pop()))

    if config['user_interface']['visualize_in_mesh']:
        o3d.visualization.draw_geometries(geometries)
        

class Ruler:
    
    def __init__(self, image_path:str, frame_name:str, disparity_map_path:str, config, point1, point2):
        
        self.visualize = config['evaluation']['measure']['visualize']
        self.path_disparity_map = disparity_map_path
        self.disparity_map_type = config['evaluation']['measure']['disparity_map_type']
        self.image_path = image_path
        self.image_path_right = image_path.replace('left', 'right')
        self.frame_name = frame_name
        self.ground_truth = config['evaluation']['measure']['ground_truth']
        self.config = config
        self.point1 = point1
        self.point2 = point2
        self.indices = None
        self.offset = config['evaluation']['measure']['offset']
        self.cp = CamPara(config)
        self.mesh_type = config['evaluation']['measure']['surface_reconstruction']
        self.every_nth = config['evaluation']['measure']['every_nth']
        self.spline_s = config['evaluation']['measure']['spline_s']
        self.path_calculation_failed = False
        self.measurement_results = {'frame_name': self.frame_name,
                                    'ground_truth': 0.0,
                    	            'euclidian': 0.0,
                                    'on_surface': 0.0,
                                    'on_surface_spline': 0.0}
        
    def get_colors_for_points(self):
    
        # Read the colors by path
        colors = cv2.imread(self.image_path, 1)
        
        colors = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)

        colors = colors[self.indices]
        
        return colors
    
    # Create a o3d marker for a given point
    def create_markers_for_points(self):
        # Create two markers
        marker1 = o3d.geometry.TriangleMesh.create_sphere(radius=1)
        marker2 = o3d.geometry.TriangleMesh.create_sphere(radius=1)

        # Translate markers to specific point in point cloud
        marker1.translate(self.point1)
        marker2.translate(self.point2)

        # Assign the color to the vertex_colors property of the marker
        marker1.paint_uniform_color([1, 0, 0])
        marker2.paint_uniform_color([1, 0, 0])

        return marker1, marker2
    
    def sgbm_disparity_calculation(self):
        
        left_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        right_image = cv2.imread(self.image_path_right, cv2.IMREAD_GRAYSCALE)

        # Create StereoSGBM matcher
        window_size = 5
        min_disp = 16
        num_disp = 112 - min_disp
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=16,
            P1=8 * 3 * window_size ** 2,
            P2=32 * 3 * window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # Calculate the disparity map
        disparity_map = stereo.compute(left_image, right_image).astype(np.float32) / 16.0
        
        return disparity_map
    
    # Loads disparity map as numpy array from path
    def get_disparity_map(self):
        
        if self.disparity_map_type == 'raft_stereo':
            self.disparity_map = np.load(self.path_disparity_map)
        elif self.disparity_map_type == 'sgbm':
            self.disparity_map = self.sgbm_disparity_calculation()
                    
    # Reproject the pixel coordinates to world coordinates using the open cv built in method.
    def reproject_to_world(self):
        return cv2.reprojectImageTo3D(self.disparity_map, self.cp.get_q()), cv2.reprojectImageTo3D(self.disparity_map, self.cp.get_q()).reshape(int(self.cp.imx*self.cp.imy), 3)
    
    # Method for cropping point cloud section (are between measurement points) to speed up meshing process
    def crop_point_cloud_by_selected_points(self, point_cloud):
        
        # Define bounding box corners
        xmin = min(self.point1[0], self.point2[0]) - self.offset
        xmax = max(self.point1[0], self.point2[0]) + self.offset
        ymin = min(self.point1[1], self.point2[1]) - self.offset
        ymax = max(self.point1[1], self.point2[1]) + self.offset
        zmin = float('-inf')
        zmax = float('inf')

        # Extract points within bounding box
        indices = np.where((point_cloud[:,:,0] > xmin) & (point_cloud[:,:,0] < xmax) &
                        (point_cloud[:,:,1] > ymin) & (point_cloud[:,:,1] < ymax) &
                        (point_cloud[:,:,2] > zmin) & (point_cloud[:,:,2] < zmax))
        
        # Filter point cloud by indices
        filtered_point_cloud = point_cloud[indices]
        
        # Return the filtered (cropped) point cloud and the indices
        return filtered_point_cloud, indices

    def pointcloud2mesh(self):
        
        if self.mesh_type == 'poisson':
            
            # Create mesh with poisson
            self.pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # Invalidate existing normals
            self.pcd.estimate_normals()  # Estimate normals
            self.mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(self.pcd,
                                                                                             depth=self.config['evaluation']['measure']['poisson']['depth'],
                                                                                             width=self.config['evaluation']['measure']['poisson']['width'],
                                                                                             scale=self.config['evaluation']['measure']['poisson']['scale'],
                                                                                             linear_fit=self.config['evaluation']['measure']['poisson']['linear_fit'])
            self.mesh.compute_vertex_normals()

            if self.config['evaluation']['measure']['poisson']['densities_optimization']:
                densities = np.asarray(densities)
                density_colors = plt.get_cmap('plasma')(
                    (densities - densities.min()) / (densities.max() - densities.min()))
                density_colors = density_colors[:, :3]
                density_mesh = o3d.geometry.TriangleMesh()
                density_mesh.vertices = self.mesh.vertices
                density_mesh.triangles = self.mesh.triangles
                density_mesh.triangle_normals = self.mesh.triangle_normals
                density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
                o3d.visualization.draw_geometries([density_mesh])

                vertices_to_remove = densities < np.quantile(densities, 0.01)
                self.mesh.remove_vertices_by_mask(vertices_to_remove)

        elif self.mesh_type == 'pivoting':
            # Create mesh with pivoting balls
            self.pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # Invalidate existing normals
            self.pcd.estimate_normals()  # Estimate normals
            radii = [0.005, 0.01, 0.02, 0.04]
            self.mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(self.pcd, o3d.utility.DoubleVector(radii))
            self.mesh.compute_vertex_normals()

        elif self.mesh_type == 'alpha':
            self.pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # Invalidate existing normals
            self.pcd.estimate_normals()
            tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(self.pcd)
            alpha = 6
            print(f"alpha={alpha:.3f}")
            self.mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(self.pcd, alpha, tetra_mesh, pt_map)
            self.mesh.compute_vertex_normals()

        else:
            print('No surface reconstruction choosen!')
    
    def compute_shortest_path(self):
        # Convert mesh to a graph
        vertices = np.asarray(self.mesh.vertices)
        faces = np.asarray(self.mesh.triangles)
        graph = nx.Graph()
        graph.add_nodes_from(range(vertices.shape[0]))
        for face in faces:
            graph.add_edge(face[0], face[1], weight=np.linalg.norm(vertices[face[0]] - vertices[face[1]]))
            graph.add_edge(face[1], face[2], weight=np.linalg.norm(vertices[face[1]] - vertices[face[2]]))
            graph.add_edge(face[2], face[0], weight=np.linalg.norm(vertices[face[2]] - vertices[face[0]]))

        # Find the closest vertices to the source and target points
        source_vertex = np.argmin(np.sum((vertices - self.point1)**2, axis=1))
        target_vertex = np.argmin(np.sum((vertices - self.point2)**2, axis=1))
        try:
            # Use Dijkstra's algorithm to find the shortest path between the two vertices
            self.path = nx.dijkstra_path(graph, source_vertex, target_vertex)
        except:
            self.path_calculation_failed = True
            
    # Compute euclidian, on surface and smoothed on surface distance
    def compute_distances(self):
        
        # Get all mesh vertices
        vertices = self.mesh.vertices
        path_points = np.array([vertices[i] for i in self.path])
        # Spline interpolation of path points
        tck, u = splprep(path_points.T, u=None, s=self.spline_s)
            
        # Simplify path by using only every_nth point
        smoothed_path_points = splev(np.linspace(0, 1, len(self.path) // self.every_nth), tck)
        smoothed_path_points = np.array(smoothed_path_points)
        path_points = path_points.transpose()
        
        # Store the distances in a dictionary 
        self.measurement_results = {'frame_name': self.frame_name,
                                    'ground_truth': self.ground_truth,
                    	            'euclidian': np.linalg.norm(path_points.transpose()[0] - path_points.transpose()[-1]),
                                    'on_surface': np.cumsum(np.sqrt(np.sum((path_points.transpose()[1:] - path_points.transpose()[:-1])**2, axis=1)))[-1],
                                    'on_surface_spline': np.cumsum(np.sqrt(np.sum((smoothed_path_points.transpose()[1:] - smoothed_path_points.transpose()[:-1])**2, axis=1)))[-1]}

        if self.visualize:
            print('Euclidian distance in mm: ', self.measurement_results['euclidian'])
            print('On-surface distance in mm: ', self.measurement_results['on_surface'])
            print('On-surface smoothed distance in mm: ', self.measurement_results['on_surface_spline'])
            self.path_points = [path_points.transpose(), smoothed_path_points.transpose(), np.array([path_points.transpose()[0],path_points.transpose()[-1]])]
    
    # Calculate the euclidian distance between two points and return it
    def get_euclidian_distance(self,):
        distance = np.linalg.norm(self.point1 - self.point2)
        self.measurement_results['euclidian'] = distance
        self.measurement_results['ground_truth'] = self.ground_truth
        
    # Create a lineset for given points and paint it in a give color
    def create_line_set(self, points, color):
        # Create a LineSet
        line_set = o3d.geometry.LineSet()
        # Add the points
        line_set.points = o3d.utility.Vector3dVector(points)
        # Create the indix combinations wich point to wich point connects
        lines = np.arange(len(points) - 1)[..., np.newaxis]
        lines = np.concatenate((lines, lines + 1), axis=1)
        # Convert the indices to integer
        lines.astype(int)
        # Add the line connections
        line_set.lines = o3d.utility.Vector2iVector(lines)
        # Add color to line set so we can identify them
        line_set.paint_uniform_color(color)
        # Return the line set
        return line_set
    
    def create_cylinder_line_set(self, points, color, radius=0.2):
        cylinders = []
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]
            length = np.linalg.norm(end - start)
            mid_point = (start + end) / 2
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius, length)
            cylinder.paint_uniform_color(color)
            direction = (end - start) / length

            # Calculate the rotation matrix to properly orient the cylinders
            axis = np.cross(np.array([0, 0, 1]), direction)
            axis /= np.linalg.norm(axis)
            angle = np.arccos(np.dot(np.array([0, 0, 1]), direction))
            rot_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

            cylinder.rotate(rot_matrix, [0, 0, 0])
            cylinder.translate(mid_point)
            cylinders.append(cylinder)
        return cylinders

    # Measure distance
    def measure(self):
    
        # Load disparity map from path
        self.get_disparity_map()
        
        # Get reprojected pixel coordinates as 3d point cloud and numpy array
        point_cloud, points = self.reproject_to_world()
        
        # Transform the pixel coordinates into world coordinates (note that the pc is in (y,x) and the pixels are in (x,y) format)
        self.point1 = point_cloud[self.point1[1], self.point1[0]]
        self.point2 = point_cloud[self.point2[1], self.point2[0]]
        
        # Crop pointcloud by relevant area (area between selected points)
        # Get cropped points and indices of cropped points
        points, self.indices = self.crop_point_cloud_by_selected_points(point_cloud=point_cloud)

        # Create an Open3D point cloud
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(points)

        if self.visualize:
            
            # Add colors of image to point cloud
            self.pcd.colors = o3d.utility.Vector3dVector(self.get_colors_for_points()/255)
            # Create a marker for each point (for visualization)
            marker1, marker2 = self.create_markers_for_points()

        # Convert pointcloud to mesh
        self.pointcloud2mesh()
        
        # Compute shortest path in mesh
        self.compute_shortest_path()
        
        if not self.path_calculation_failed:
            
            # Compute the distances
            self.compute_distances()
            
            if self.visualize:
                
                # Create a transformation matrix that flips the Z-axis
                transformation_matrix = np.array([[-1, 0, 0, 0],
                                                    [0, 1, 0, 0],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]], dtype=np.float64)
                                                                
                # Make list of geometries
                #geometries = [self.pcd, marker1, marker2]
                geometries = [self.pcd, marker1, marker2]
                #geometries = [self.pcd]
                # Colors for lines
                line_colors = [[0, 1, 1], [1, 0, 1],[0, 1, 0]]

                # create line sets and append them to list of geometries
                # for points in self.path_points:
                #     geometries.append(self.create_line_set(points=points, color=line_colors.pop()))
                
                #Create line sets and append them to the list of geometries
                for points in self.path_points:
                    cylinders = self.create_cylinder_line_set(points=points, color=line_colors.pop())
                    geometries.extend(cylinders)
                
                for geometry in geometries:
                    geometry.transform(transformation_matrix)
                
                # Draw geometries
                o3d.visualization.draw_geometries(geometries)
                
        else:
            self.get_euclidian_distance()
                
        # Return results 
        return self.measurement_results
    