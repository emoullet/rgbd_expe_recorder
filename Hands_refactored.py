import numpy as np
import pandas as pd
import trimesh as tm
import cv2
import time

from typing import List, Iterable, Tuple, Union, Optional

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from i_grip.utils2 import *
from i_grip import Hands3DDetectors as hd
from typing import  Mapping
from i_grip.Hands3DDetectors import HandPrediction
from i_grip.Plotters import Plotter
from typing import List, Union
    
class GraspingHandTrajectory(Trajectory):
    """GraspingHandTrajectory class to handle the trajectory of a grasping hand. It inherits from Trajectory.

    Attributes:
        DEFAULT_DATA_KEYS (list): Default list of data keys
        DEFAULT_ATTRIBUTES (dict): Default attributes
        poly_coeffs (dict): Dictionary of polynomial coefficients
        polynomial_function (function): Polynomial function
        was_fitted (bool): Flag to indicate if the trajectory was fitted
        

    Raises:
        StopIteration: If the end of the trajectory is reached
        IndexError: If the index is out of range

    """
    
    DEFAULT_DATA_KEYS = [ 'Timestamps', 'x', 'y', 'z', 'Extrapolated']
    DEFAULT_ATTRIBUTES  = dict(timestamp=True, filtered_position=True)
    
    def __init__(self, state: 'GraspingHandState' = None, headers_list: list = DEFAULT_DATA_KEYS, attributes_dict: dict = DEFAULT_ATTRIBUTES, file: str = None, dataframe: pd.DataFrame = None, limit_size: int = None)-> None:
        """Constructor for the GraspingHandTrajectory class. It initializes the trajectory with the provided data. 

        Args:
            state (GraspingHandState, optional): First state to initialize the trajectory. Defaults to None.
            headers_list (list, optional): List of data keys. Defaults to DEFAULT_DATA_KEYS.
            attributes_dict (dict, optional): Dictionary of attributes. Defaults to DEFAULT_ATTRIBUTES.
            file (str, optional): File to load the data from. Defaults to None.
            dataframe (pd.DataFrame, optional): Dataframe to load the data from. Defaults to None.
            limit_size (int, optional): Limit size of the trajectory. Defaults to None.
            
        """
        super().__init__(state, headers_list, attributes_dict, file, dataframe, limit_size)
        self.poly_coeffs = {}
        self.polynomial_function = None
        self.was_fitted = False
            
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, headers_list: list = DEFAULT_DATA_KEYS, attributes_dict: dict = DEFAULT_ATTRIBUTES, limit_size: int = None)-> 'GraspingHandTrajectory':
        """Create a GraspingHandTrajectory object from a dataframe.

        Args:
            df (pd.DataFrame): Dataframe containing the trajectory data.
            headers_list (list, optional): List of data keys. Defaults to DEFAULT_DATA_KEYS.
            attributes_dict (dict, optional): Dictionary of attributes. Defaults to DEFAULT_ATTRIBUTES
            limit_size (int, optional): Limit size of the trajectory. Defaults to None.
            

        Returns:
            GraspingHandTrajectory: GraspingHandTrajectory object created from the dataframe.
        """
        return cls(dataframe = df, headers_list=headers_list, attributes_dict=attributes_dict, limit_size=limit_size)
    
    def fit(self, nb_points: int = 20, degree: int = 2)-> None:
        """Fit a polynomial to the trajectory data. The polynomial is fitted to the last nb_points of the trajectory. Weights are computed based on the timestamps of the data points to give more importance to recent data points.

        Args:
            nb_points (int, optional): Number of data points to use for the polynomial fit. Defaults to 20.
            degree (int, optional): Degree of the polynomial. Defaults to 2.
            
        Raises:
            ValueError: If there are not enough data points to find a polynomial fit.
        """
        if len(self.data) < nb_points:
            # raise ValueError('Not enough data points to find a polynomial fit')
            print('Not enough data points to find a polynomial fit')
        else:
            last_points = self.data.iloc[-nb_points:]
            t = last_points['Timestamps'].values
            x = last_points['x'].values
            y = last_points['y'].values
            z = last_points['z'].values
            
            # weights = 1/(1+np.sqrt(np.abs(t)))
            weights = 1/(1+np.abs(t))
            
            self.poly_coeffs['x'] = np.polynomial.polynomial.Polynomial.fit(t, x, degree, w=weights)
            self.poly_coeffs['y'] = np.polynomial.polynomial.Polynomial.fit(t, y, degree, w=weights)
            self.poly_coeffs['z'] = np.polynomial.polynomial.Polynomial.fit(t, z, degree, w=weights)
            self.was_fitted = True

    def extrapolate(self, timestamps:Iterable)-> np.array:
        """Extrapolate the trajectory to the given timestamps using the fitted polynomial.

        Args:
            timestamps (Iterable): Timestamps to extrapolate the trajectory to.

        Returns:
            np.array: Extrapolated trajectory points.
        """
        if not self.was_fitted:
            x = self.data.loc[len(self.data)-1]['x']
            y = self.data.loc[len(self.data)-1]['y']
            z = self.data.loc[len(self.data)-1]['z']
            return np.array([x,y,z])
        else:
            x = self.poly_coeffs['x'](np.array(timestamps))
            y = self.poly_coeffs['y'](np.array(timestamps))
            z = self.poly_coeffs['z'](np.array(timestamps))
            res = np.array([x,y,z]).T
        return res

    
    def compute_last_derivatives(self, nb_points:int=None)-> Tuple[np.array, np.array]:
        """Compute the velocity and acceleration of the trajectory at the last data point. The derivatives are computed using finite differences.

        Args:
            nb_points (int, optional): The number of data points to use to compute the derivatives. If None, all data points are used. Defaults to None.

        Returns:
            Tuple[np.array, np.array]: Velocity and acceleration of the trajectory at the last data point.
        """
        if nb_points is None:
            print('Using all data points to compute derivatives')
            last_points = self.data
        elif len(self.data) < nb_points:
            print(f'Not enough data points to compute derivatives, using all data points ({len(self.data)})')
            last_points = self.data
        else:
            last_points = self.data.iloc[-nb_points:]
        x = last_points['x'].values
        y = last_points['y'].values
        z = last_points['z'].values
        t = last_points['Timestamps'].values
        max_acc = 2
        t = np.array(t)
        if len(t) < 2:
            velocity = np.array([0,0,0])
            acceleration = np.array([0,0,0])
        elif len(t) < 3:
            velocity = np.array([x[1]-x[0], y[1]-y[0], z[1]-z[0]])/(t[1]-t[0])
            acceleration = np.array([0,0,0])
        else:
            positions = np.array([x,y,z]).T
            if len(t) <= 5:
                acc = 2
            elif len(t) <= 8:
                acc = 4
            elif len(t) <= 12:
                acc = 6
            else:
                acc = 8
            acc = min(acc, max_acc)
            velocity = findiff_diff(t, positions, diff_order=1, diff_acc=acc)
            acceleration = np.array([0,0,0])
        return velocity, acceleration
    
    def get_xyz_data(self)-> Tuple[List[np.array], List[np.array]]:
        """Return the observed and extrapolated xyz data points.

        Returns:
            Tuple[List[np.array], List[np.array]]: XYZ observed and extrapolated data points.
        """
        xyz_observed_list = []
        xyz_extrapolated_list = []
        
        last_points = self.data.iloc[-100:]
        # print(f'last_points : {last_points}')
        for index, row in last_points.iterrows():
            xyz = np.array([-row['x'], row['y'], row['z']])
            if row['Extrapolated']:
                xyz_extrapolated_list.append(xyz)
            else:                
                xyz_observed_list.append(xyz)
        return xyz_observed_list, xyz_extrapolated_list
    
    
    def __next__(self):
        if self.current_line_index < len(self.data):
            row = self.data.iloc[self.current_line_index]
            # print(f'data : {self.data}')
            # print('row', row)
            self.current_line_index+=1
            return Position(np.array([row['x'], row['y'], row['z']]), display='cm', swap_y=True), row['Timestamps']
        else:
            raise StopIteration
        
    def __getitem__(self, index):
        if index < len(self.data):
            row = self.data.iloc[index]
            return Position(np.array([row['x'], row['y'], row['z']]), display='cm', swap_y=True), row['Timestamps']
        else:
            raise IndexError('Index out of range')
    
    def __repr__(self) -> str:
        return super().__repr__()


class GraspingHand(Entity):
    """GraspingHand class to handle a grasping hand. It inherits from Entity. A grasping hand is defined by its state, trajectory, and label.

    Attributes:
        MAIN_DATA_KEYS (list): Main data keys
        label (str): Label of the hand
        plotter (Plotter): Plotter object
        detected_hand (HandPrediction): Detected hand
        state (GraspingHandState): State of the hand
        new (bool): Flag to indicate if the hand is new
        visible (bool): Flag to indicate if the hand is visible
        invisible_time (float): Time the hand has been invisible
        max_invisible_time (float): Maximum time the hand can be invisible
        show_label (bool): Flag to indicate if the label is displayed
        show_xyz (bool): Flag to indicate if the xyz coordinates are displayed
        show_roi (bool): Flag to indicate if the region of interest is displayed
        margin (int): Margin for the displayed text
        font_size (int): Font size for the displayed text
        font_thickness (int): Font thickness for the displayed text
        label_text_color (tuple): Label text color
        font_size_xyz (float): Font size for the xyz coordinates
        impact_locations (dict): Dictionary of impact locations
        hand_pos_obj_frame (dict): Dictionary of hand positions in the object frame
        impact_locations_list (dict): Dictionary of impact locations list
        rays_vizualize_list (dict): Dictionary of rays to visualize
        full_hand (bool): Flag to indicate if the full hand is displayed
        mesh_origin (trimesh.primitives.Sphere): Mesh origin
        impact_color (list): Impact color
        mesh_key_points (list): List of mesh key points
        key_points_key_points_connections (list): List of key points connections
        key_points_connections_path (trimesh.path.Path): Key points connections path
        mesh_position (Position): Mesh position
        mesh_transform (np.array): Mesh transform
        most_probable_target (str): Most probable target
        targets_data (pd.DataFrame): Targets data
    """
    MAIN_DATA_KEYS=GraspingHandTrajectory.DEFAULT_DATA_KEYS
    
    def __init__(self, input: Union[hd.HandPrediction, np.ndarray, pd.DataFrame], label: Optional[str] = None, timestamp: Optional[float] = None, plotter: Optional[Plotter] = None) -> None:
        """
        Initializes a new instance of the Hands class.
        Parameters:
        - input: Union[hd.HandPrediction, np.ndarray, pd.DataFrame]
            The input data to initialize the Hands object. It can be one of the following types:
            - hd.HandPrediction: A hand prediction object.
            - np.ndarray: An array representing the hand position.
            - pd.DataFrame: A dataframe containing hand data.
        - label: Optional[str]
            The label associated with the hand.
        - timestamp: Optional[float]
            The timestamp of the hand data.
        - plotter: Optional[Plotter]
            The plotter object used for visualization.
        Raises:
        - ValueError: If input is None.
        """
        super().__init__(timestamp=timestamp)
        self.label = label
        self.plotter = plotter
        if input is None:
            raise ValueError('input must be provided')
        else:
            if isinstance(input, hd.HandPrediction):
                self.detected_hand = input
                self.state = GraspingHandState.from_hand_detection(input, timestamp = timestamp)
                if self.label is None:
                    self.label = input.label         
            elif isinstance(input, np.ndarray):
                self.state = GraspingHandState.from_position(input, timestamp = timestamp)            
            elif isinstance(input, pd.DataFrame):
                self.state = GraspingHandState.from_dataframe(input)
                
            else:   
                self.state = None
        
        self.new= True
        self.visible = True
        self.invisible_time = 0
        self.max_invisible_time = 0.3
        
        self.show_label = True
        self.show_xyz = True
        self.show_roi = True
        self.margin = 10  # pixels
        self.font_size = 1
        self.font_thickness = 1
        self.label_text_color = (88, 205, 54) # vibrant green
        self.font_size_xyz = 0.5
    
        ### define mesh
        self.impact_locations = {}
        self.hand_pos_obj_frame = {}
        self.impact_locations_list = {}
        self.rays_vizualize_list = {}
        self.full_hand = False
        self.define_mesh_representation()
        self.update_mesh()
        self.mesh_position = Position(self.state.position_filtered*np.array([-1,1,1]))
        self.mesh_transform= tm.transformations.translation_matrix(self.mesh_position.v)
        self.most_probable_target = None
        self.targets_data = pd.DataFrame(columns = ['Timestamp', 'label', 'grip', 'time_before_impact','ratio'])
        self.future_points_from_target_detector=[]

    @classmethod
    def from_trajectory(cls, trajectory):
        return cls(trajectory = trajectory)
    
    def build_from_scratch(self, label):        
        self.label = label
        self.state = GraspingHandState()
        
    def define_mesh_representation(self):
        """
        Defines the mesh representation for the hand object.
        This method sets the mesh origin, impact color, and various colors based on the label of the hand.
        If the label is 'right', the mesh origin and colors are set accordingly.
        If the label is not 'right', the mesh origin and colors are set differently.
        If the full_hand flag is True, additional mesh key points are created and their colors are set.
        Returns:
            None
        """
        pass
        self.mesh_origin = tm.primitives.Sphere(radius = 20)
        self.impact_color = [255,0,255,255]
        if self.label == 'right':
            self.mesh_origin.visual.face_colors = self.color = [255,0,0,255]
            self.plot_color = 'red'
            self.extrapolated_trajectory_color = [0,100,0,100]
            self.future_color = [0,255,0,255]
            self.text_color = [0,0,255]
        else:  
            self.mesh_origin.visual.face_colors = self.color = [0,0,100,255]
            self.extrapolated_trajectory_color = [50,50,50,50]
            self.text_color=[100,0,0]
            self.plot_color = 'blue'
            self.future_color = [0,255,255,255]
            
        if self.full_hand:
            self.mesh_key_points = [tm.primitives.Sphere(radius = 7) for i in range(21)]
            mp_specs = solutions.drawing_styles.get_default_hand_landmarks_style()
            for i in range(21):                
                drawing_spec = mp_specs[i] if isinstance(
                    mp_specs, Mapping) else mp_specs
                color = drawing_spec.color
                self.mesh_key_points[i].visual.face_colors = [color[0], color[1], color[2], 255]
            self.key_points_key_points_connections=solutions.hands.HAND_CONNECTIONS
            self.key_points_connections_path = tm.load_path(np.zeros((2,2,3)))
    
    def get_keypoints_representation(self):
        return self.key_points_mesh_transforms, self.key_points_connections_path
    
    def update(self, detected_hand: hd.HandPrediction) -> None:
        """
        Updates the detected hand with the given hand prediction.

        Parameters:
            detected_hand (hd.HandPrediction): The hand prediction to update with.

        Returns:
            None
        """
        self.detected_hand = detected_hand
        self.state.update(detected_hand)
        
    def update_from_trajectory(self, index = None):
        if index is None:         
            pos, timestamp= self.state.__next__()
        else:
            pos, timestamp = self.state[index]
        self.state.update(pos)
        self.state.propagate_all(timestamp)
        self.set_mesh_updated(False)

    def propagate(self, timestamp:float=None):
        """
        Propagates the state and updates the mesh position and transform.

        Args:
            timestamp (float, optional): The timestamp for propagation. Defaults to None.
        """
        self.state.propagate_all(timestamp)
        self.set_mesh_updated(False)
        self.mesh_position = Position(self.state.position_filtered*np.array([-1,1,1]))
        self.mesh_transform= tm.transformations.translation_matrix(self.mesh_position.v)
              
    def update_mesh(self):
        """
        Update the mesh based on the current state.
        Returns:
            None
        """
        
        if  self.was_mesh_updated():
            return
        if self.full_hand:
            self.key_points_mesh_transforms = [tm.transformations.translation_matrix(self.state.world_landmarks[i]*np.array([-1,1,1])) for i in range(21)]
            self.key_points_connections_path_starts = np.vstack([self.state.world_landmarks[connection[0]]*np.array([-1,1,1]) for connection in self.key_points_key_points_connections])
            self.key_points_connections_path_ends = np.vstack([self.state.world_landmarks[connection[1]]*np.array([-1,1,1]) for connection in self.key_points_key_points_connections])
            self.key_points_connections_path = tm.load_path(np.hstack((self.key_points_connections_path_starts, self.key_points_connections_path_ends)).reshape(-1, 2, 3)) 
        self.set_mesh_updated(True)
    
    def get_keypoints_representation(self):
        return self.key_points_mesh_transforms, self.key_points_connections_path
    
    def get_mesh_position(self):
        return self.mesh_position.v
    
    def get_trajectory_points(self):
        return self.state.trajectory.get_xyz_data()
    
    def get_movement_direction(self):
        return self.state.get_movement_direction()
    
    def get_scalar_velocity(self):
        return self.state.scalar_velocity
    
    def get_future_trajectory_points(self):
        self.future_points = self.state.get_future_trajectory_points()
        return self.future_points
    
    def set_future_points_from_target_detector(self, future_points):
        self.future_points_from_target_detector = future_points
    
    def get_future_points_from_target_detector(self):
        return self.future_points_from_target_detector

    def render(self, img: np.array) -> None:
        """
        Renders the hand on the given image.
        Args:
            img (np.array): The image on which to render the hand.
        Returns:
            None
        """
        
        GraspingHand.render_hand(img, self.label, self.detected_hand.get_landmarks(), self.detected_hand.roi, self.state.position_filtered, self.show_label, self.show_xyz, self.show_roi, self.margin, self.font_size, self.font_thickness, self.font_size_xyz, self.label_text_color)
    
    def render_hand(img: np.array, label: Optional[str] = None, displayed_landmarks: Optional[np.ndarray] = None, roi: Optional[Tuple[int, int, int, int]] = None, position: Optional[Position] = None, show_label: bool = True, show_xyz: bool = True, show_roi: bool = True, margin: int = 10, font_size: int = 1, font_thickness: int = 1, font_size_xyz: float = 0.5, label_text_color: Tuple[int, int, int] = (88, 205, 54)) -> None:
        """
        Renders a hand on the given image with optional label, displayed landmarks, region of interest (ROI), and position information.
        Parameters:
        - img: The input image as a numpy array.
        - label: Optional label for the hand.
        - displayed_landmarks: Optional array of landmarks to be displayed on the hand.
        - roi: Optional region of interest (ROI) as a tuple of (x1, y1, x2, y2) coordinates.
        - position: Optional position information of the hand.
        - show_label: Whether to show the label on the image.
        - show_xyz: Whether to show the X, Y, Z position information on the image.
        - show_roi: Whether to show the ROI on the image.
        - margin: Margin value for positioning the label and XYZ information.
        - font_size: Font size for the label.
        - font_thickness: Font thickness for the label.
        - font_size_xyz: Font size for the XYZ information.
        - label_text_color: Text color for the label.
        Returns:
        None
        """
        
        if displayed_landmarks is not None:
            # Draw the hand landmarks.
            landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            landmarks_proto.landmark.extend([
                                                landmark_pb2.NormalizedLandmark(x=landmark[0], y=landmark[1], z=landmark[2]) for landmark in displayed_landmarks])
            solutions.drawing_utils.draw_landmarks(
                                                img,
                                                landmarks_proto,
                                                solutions.hands.HAND_CONNECTIONS,
                                                solutions.drawing_styles.get_default_hand_landmarks_style(),
                                                solutions.drawing_styles.get_default_hand_connections_style())
        if show_label:
            # Get the top left corner of the detected hand's bounding box.
            text_x = int(min(displayed_landmarks[:,0]))
            text_y = int(min(displayed_landmarks[:,1])) - margin

            # Draw handedness (left or right hand) on the image.
            cv2.putText(img, f"{label}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        font_size, label_text_color, font_thickness, cv2.LINE_AA)
        if show_roi and roi is not None:
            cv2.rectangle(img, (roi[0],roi[1]),(roi[2],roi[3]),label_text_color)

        if show_xyz and position is not None:
            # Get th e top left corner of the detected hand's bounding box.z
            
            #print(f"{label} --- X: {xyz[0]/10:3.0f}cm, Y: {xyz[0]/10:3.0f} cm, Z: {xyz[0]/10:3.0f} cm")
            if len(img.shape)<3:
                height, width = img.shape
            else:
                height, width, _ = img.shape
            x_coordinates = displayed_landmarks[:,0]
            y_coordinates = displayed_landmarks[:,1]
            x0 = int(max(x_coordinates) * width)
            y0 = int(max(y_coordinates) * height) + margin

            # Draw handedness (left or right hand) on the image.
            cv2.putText(img, f"X:{position.x/10:3.0f} cm", (x0+10, y0+20), cv2.FONT_HERSHEY_DUPLEX, font_size_xyz, (20,180,0), font_thickness, cv2.LINE_AA)
            cv2.putText(img, f"Y:{position.y/10:3.0f} cm", (x0+10, y0+45), cv2.FONT_HERSHEY_DUPLEX, font_size_xyz, (255,0,0), font_thickness, cv2.LINE_AA)
            cv2.putText(img, f"Z:{position.z/10:3.0f} cm", (x0+10, y0+70), cv2.FONT_HERSHEY_DUPLEX, font_size_xyz, (0,0,255), font_thickness, cv2.LINE_AA)
    
    def get_rendering_data(self)-> dict:
        """
        Returns the rendering data for the hand.
        Returns:
            dict: A dictionary containing the following information:
                - 'displayed_landmarks': The landmarks of the detected hand.
                - 'roi': The region of interest of the detected hand.
                - 'position': The filtered position of the hand.
                - 'font_size': The font size for rendering.
                - 'font_thickness': The font thickness for rendering.
                - 'font_size_xyz': The font size for rendering XYZ coordinates.
                - 'label_text_color': The text color for rendering labels.
                - 'label': The label for the hand.
                - 'show_label': A flag indicating whether to show the label.
                - 'show_xyz': A flag indicating whether to show XYZ coordinates.
                - 'show_roi': A flag indicating whether to show the region of interest.
                - 'margin': The margin for rendering.
        """
        
        data={}
        data['displayed_landmarks'] = self.detected_hand.get_landmarks()
        data['roi'] = self.detected_hand.roi
        data['position'] = self.state.position_filtered
        data['font_size'] = self.font_size
        data['font_thickness'] = self.font_thickness
        data['font_size_xyz'] = self.font_size_xyz
        data['label_text_color'] = self.label_text_color
        data['label'] = self.label
        data['show_label'] = self.show_label
        data['show_xyz'] = self.show_xyz
        data['show_roi'] = self.show_roi
        data['margin'] = self.margin
        return data

    def get_target_data(self):
        return self.targets_data
        
class GraspingHandState(State):
    def __init__(self,  position:Position = None, normalized_landmarks:np.array = None, world_landmarks:np.array = None, timestamp:float = None, trajectory:GraspingHandTrajectory = None):
        """
        Initializes a new instance of the GraspHandState class.
        Args:
            position (Position, optional): The position of the hand. Defaults to None.
            normalized_landmarks (np.array, optional): The normalized landmarks of the hand. Defaults to None.
            world_landmarks (np.array, optional): The world landmarks of the hand. Defaults to None.
            timestamp (float, optional): The timestamp of the hand. Defaults to None.
            trajectory (GraspingHandTrajectory, optional): The trajectory of the hand. Defaults to None.
        """
        
        super().__init__()  # Call the parent class constructor
        
        # Initialize instance variables
        self.position_raw = Position(position)
        self.normalized_landmarks = normalized_landmarks
        self.world_landmarks = world_landmarks
        self.new_position = self.position_raw
        self.new_normalized_landmarks = self.normalized_landmarks
        self.new_world_landmarks = self.world_landmarks
        
        self.velocity_raw = np.array([0,0,0])
        self.scalar_velocity = 0
        self.normed_velocity = np.array([0,0,0])
        self.position_filtered = self.position_raw
        self.velocity_filtered = self.velocity_raw
        
        # Create filters for position and velocity
        self.filter_position, self.filter_velocity = Filter.both('position')
        
        if normalized_landmarks is not None:
            self.normalized_landmarks_velocity = np.zeros((21,3))
            self.normalized_landmarks_filtered = self.normalized_landmarks
            self.normalized_landmarks_velocity_filtered = self.normalized_landmarks_velocity            
            # Create filters for normalized landmarks and their velocity
            self.filter_normalized_landmarks, self.filter_normalized_landmarks_velocity= Filter.both('normalized_landmarks')
            
        self.future_points=[self.position_raw.v]
            
        if timestamp is None:
            self.last_timestamp = time.time()
        else:
            self.last_timestamp = timestamp
        self.scalar_velocity_threshold = 20 #mm/s
        
        if trajectory is None:
            self.trajectory = GraspingHandTrajectory.from_state(self)
        else:
            self.trajectory = trajectory
        
        self.extrapolation_count = 0
        self.set_updated(True)
        self.set_propagated(False)
        
    @classmethod
    def from_hand_detection(cls, hand_detection: hd.HandPrediction, timestamp = 0):
        return cls(hand_detection.position, hand_detection.normalized_landmarks, hand_detection.world_landmarks, timestamp)
    
    @classmethod
    def from_position(cls, position: Position, timestamp = 0):
        return cls(position, timestamp=timestamp)

    @classmethod
    def from_dataframe(cls, df:pd.DataFrame):
        trajectory = GraspingHandTrajectory.from_dataframe(df)
        first_position, first_timestamp = trajectory[0]
        return cls(first_position, timestamp=first_timestamp, trajectory=trajectory)
    
    def update_position(self, position):
        self.new_position = Position(position)
        
    def update_normalized_landmarks(self, normalized_landmarks):
        self.new_normalized_landmarks = normalized_landmarks
    
    def update_world_landmarks(self, world_landmarks):
        self.new_world_landmarks = world_landmarks
        
    def check_new_position(self, elapsed ):   
        check = True
        if Position.distance(self.position_raw, self.new_position)/elapsed > 1000:
            check = False
        return check
    
    def update(self, new_input: Union[hd.HandPrediction, Position, None])-> None:
        """
        Updates the state of the object based on the new input.
        Parameters:
        - new_input (Union[hd.HandPrediction, Position, None]): The new input to update the object's state.
        Returns:
        - None
        """
        
        if isinstance(new_input, hd.HandPrediction):            
            self.update_position(new_input.position)
            self.update_normalized_landmarks(new_input.normalized_landmarks)
            self.update_world_landmarks(new_input.world_landmarks)
        elif isinstance(new_input, Position):
            self.update_position(new_input)
        elif new_input is None:
            self.update_normalized_landmarks(new_input)
        else:
            print(f'weird input : {new_input}')
            return
        self.set_updated(True)
        self.set_propagated(False)

        
    def propagate_position(self, elapsed:float)-> None:
        """
        Propagates the position of the object based on the elapsed time.
        Args:
            elapsed (float): The elapsed time since the last update.
        Returns:
            None
        """
        
        if not self.was_updated():
            # If the state was not updated, use the current position as the next position
            next_position = self.position_raw
        else:
            # If the state was updated, use the new position as the next position
            next_position = self.new_position
            
            # Apply position filter to the next position
            next_position_filtered = Position(self.filter_position.apply(next_position.v))
            
            if elapsed > 0:
                # Compute the velocity based on the change in position and elapsed time
                self.velocity_raw = (next_position_filtered.v - self.position_filtered.v) / elapsed
            
            # Update the filtered position with the next position
            self.position_filtered = next_position_filtered
            
            # Apply velocity filter to the raw velocity
            self.velocity_filtered = self.filter_velocity.apply(self.velocity_raw)
            
            # Compute the scalar velocity
            self.scalar_velocity = np.linalg.norm(self.velocity_filtered)
            
            if self.scalar_velocity != 0:
                if self.scalar_velocity > self.scalar_velocity_threshold:
                    # If the scalar velocity is above the threshold, update the normalized velocity
                    self.normed_velocity = self.velocity_filtered / self.scalar_velocity
                else:
                    # If the scalar velocity is below the threshold, update the normalized velocity with a weighted average
                    self.normed_velocity = self.normed_velocity * 98/100 + self.velocity_filtered / self.scalar_velocity * 2/100
            else:
                # If the scalar velocity is zero, set the normalized velocity to zero
                self.normed_velocity = np.array([0, 0, 0])
            
            # Update the raw position with the next position
            self.position_raw = next_position
    
    def propagate_all(self, timestamp:float)-> None:       
        """
        Propagates the hand state by updating the position, applying filters,
        computing velocity, propagating normalized landmarks and world landmarks,
        computing future points, and setting the updated and propagated flags.

        Parameters:
        - timestamp (float): The current timestamp.

        Returns:
        - None
        """
        # Calculate the elapsed time since the last update
        elapsed = timestamp - self.last_timestamp
        
        # Update the last timestamp to the current timestamp
        self.last_timestamp = timestamp
        
        # Propagate only the position based on the elapsed time
        self.propagate_only_position()
        
        # Apply the position filter to the raw position
        self.apply_filter_position()
        
        # Compute the velocity based on the position change
        self.compute_velocity()
        
        # Propagate the normalized landmarks if available
        if self.normalized_landmarks is not None:
            self.propagate_normalized_landmarks(elapsed)
        
        # Propagate the world landmarks if available
        if self.world_landmarks is not None:
            self.propagate_world_landmarks(elapsed) 
        
        # Compute the future points based on the trajectory
        self.compute_future_points()
        
        # Set the updated and propagated flags
        self.set_updated(False)
        self.set_propagated(True)
    
    def propagate_only_position(self):
        """
        Propagates the position of an object based on its trajectory.

        If the object was updated, the raw position is set to the new position,
        the object is added to the trajectory, and the trajectory is fitted with
        a maximum of 20 points. The extrapolation count is reset to 0.

        If the object was not updated but extrapolation is allowed, the position
        is extrapolated using the last timestamp and set as the raw position. The
        object is added to the trajectory as an extrapolated point, and the
        extrapolation count is incremented.

        Returns:
            None
        """
        # Check if the state was updated
        if self.was_updated():                   
            # If updated, set the raw position to the new position
            self.position_raw = self.new_position
            # Add the state to the trajectory
            self.trajectory.add(self)
            # Fit the trajectory with a maximum of 20 points
            self.trajectory.fit(20)
            # Reset the extrapolation count
            self.extrapolation_count = 0
        # If the state was not updated but extrapolation is allowed
        elif self.extrapolation_allowed():
            # Extrapolate the position using the last timestamp
            extrapolated_position = self.trajectory.extrapolate(self.last_timestamp)
            # Set the raw position to the extrapolated position
            self.position_raw = Position(extrapolated_position)
            # Add the state to the trajectory as an extrapolated point
            self.trajectory.add(self, extrapolated=True)
            # Increment the extrapolation count
            self.extrapolation_count += 1
            
    def extrapolation_allowed(self):
        return self.extrapolation_count < 5
            
    def compute_future_points(self, nb_steps: int = 10, timestep: float = 0.1) -> None:
        """
        Computes the future points of the object based on the trajectory.

        Parameters:
        - nb_steps (int): The number of steps to compute the future points.
        - timestep (float): The time interval between each step.

        Returns:
        - None
        """
        # Compute the number of steps based on the scalar velocity and velocity unit
        vel_unit: int = 50
        nb_steps = int(self.scalar_velocity / vel_unit)
        
        # Compute the timestamps for each step
        timestamps = [self.last_timestamp + timestep * (i + 1) for i in range(nb_steps)]
        
        # Initialize the future points list
        self.future_points = []
        
        # If there are no steps, add the filtered position as a future point
        if nb_steps == 0:
            self.future_points.append(self.position_filtered.v * np.array([-1, 1, 1]))
        
        # Extrapolate the future points based on the timestamps
        future_points = self.trajectory.extrapolate(timestamps)
        
        # Add the extrapolated future points to the future points list
        self.future_points += [future_point * np.array([-1, 1, 1]) for future_point in future_points]
            
    def get_future_trajectory_points(self):
        return self.future_points

    def apply_filter_position(self):
        self.position_filtered = Position(self.filter_position.apply(self.position_raw.v))
            
    def compute_velocity(self):
        """
        Computes the velocity of the object based on its trajectory.

        If the state was not updated, the velocity is not computed.
        Otherwise, the raw velocity and acceleration are computed using the trajectory.
        The raw velocity is then filtered, and the scalar velocity is calculated.
        If the scalar velocity is not zero, the normalized velocity is computed.
        If the scalar velocity is greater than the velocity threshold, the normalized velocity is set to the filtered velocity.
        Otherwise, the normalized velocity is a weighted average of the previous normalized velocity and the filtered velocity.

        Returns:
            None
        """
        if not self.was_updated():
            return

        # Compute the raw velocity and acceleration using the trajectory
        self.velocity_raw, self.acceleration_raw = self.trajectory.compute_last_derivatives(2)

        # Filter the raw velocity
        self.velocity_filtered = self.filter_velocity.apply(self.velocity_raw)

        # Calculate the scalar velocity
        self.scalar_velocity = np.linalg.norm(self.velocity_filtered)

        if self.scalar_velocity != 0:
            if self.scalar_velocity > self.scalar_velocity_threshold:
                # Set the normalized velocity to the filtered velocity
                self.normed_velocity = self.velocity_filtered / self.scalar_velocity
            else:
                # Compute the weighted average of the previous normalized velocity and the filtered velocity
                self.normed_velocity = self.normed_velocity * 98 / 100 + self.velocity_filtered / self.scalar_velocity * 2 / 100
        else:
            # Set the normalized velocity to zero
            self.normed_velocity = np.array([0, 0, 0])
            
    
        
    def propagate_normalized_landmarks(self, elapsed: float) -> None:
        """
        Propagates the normalized landmarks of the hand state.

        If the state was not updated, the next normalized landmarks are set to the current normalized landmarks,
        and the normalized landmarks velocity remains unchanged.
        Otherwise, the next normalized landmarks are set to the new normalized landmarks,
        and the normalized landmarks velocity is computed based on the change in normalized landmarks and elapsed time.

        The next normalized landmarks and normalized landmarks velocity are then filtered using their respective filters.

        Parameters:
        - elapsed (float): The elapsed time since the last update.

        Returns:
        - None
        """
        # Check if the state was updated
        if not self.was_updated():
            # If not updated, set the next normalized landmarks to the current normalized landmarks
            next_normalized_landmarks = self.normalized_landmarks
            # Set the normalized landmarks velocity to the current normalized landmarks velocity
            self.normalized_landmarks_velocity = self.normalized_landmarks_velocity
        else:
            # If updated, set the next normalized landmarks to the new normalized landmarks
            next_normalized_landmarks = self.new_normalized_landmarks
            # Calculate the normalized landmarks velocity based on the change in normalized landmarks and elapsed time
            if elapsed > 0:
                self.normalized_landmarks_velocity = (self.new_normalized_landmarks - self.normalized_landmarks) / elapsed

        # Apply the filter to the next normalized landmarks
        self.normalized_landmarks_filtered = self.filter_normalized_landmarks.apply(next_normalized_landmarks)
        # Apply the filter to the normalized landmarks velocity
        self.normalized_landmarks_velocity_filtered = self.filter_normalized_landmarks_velocity.apply(
            self.normalized_landmarks_velocity)

        # Update the normalized landmarks to the next normalized landmarks
        self.normalized_landmarks = next_normalized_landmarks
    
    def propagate_world_landmarks(self, elapsed):
        self.world_landmarks = self.new_world_landmarks
    
    def get_movement_direction(self):
        """
        Calculates the movement direction of the hand state.

        The movement direction is calculated by adding a downward factor to the normalized velocity,
        normalizing the resulting vector, and returning it.

        Returns:
        - vdir (np.array): The movement direction vector.
        """
        # Define the downward factor
        point_down_factor = -0.3

        # Calculate the movement direction vector by adding the downward factor to the normalized velocity
        vdir = self.normed_velocity + np.array([0, point_down_factor, 0])

        # Normalize the movement direction vector
        if np.linalg.norm(vdir) != 0:
            vdir = vdir / np.linalg.norm(vdir)

        return vdir

    def as_list(self, timestamp: bool = True, position: bool = False, normalized_landmarks: bool = False, velocity: bool = False, normalized_landmarks_velocity: bool = False, filtered_position: bool = False, filtered_velocity: bool = False, filtered_normalized_landmarks: bool = False, filtered_normalized_landmarks_velocity: bool = False, normalized_velocity: bool = False, scalar_velocity: bool = False) -> List[Union[float, List[float]]]:    
        """
        Returns a list representation of the hand state.

        Parameters:
        - timestamp (bool): Whether to include the timestamp in the list.
        - position (bool): Whether to include the raw position in the list.
        - normalized_landmarks (bool): Whether to include the normalized landmarks in the list.
        - velocity (bool): Whether to include the raw velocity in the list.
        - normalized_landmarks_velocity (bool): Whether to include the normalized landmarks velocity in the list.
        - filtered_position (bool): Whether to include the filtered position in the list.
        - filtered_velocity (bool): Whether to include the filtered velocity in the list.
        - filtered_normalized_landmarks (bool): Whether to include the filtered normalized landmarks in the list.
        - filtered_normalized_landmarks_velocity (bool): Whether to include the filtered normalized landmarks velocity in the list.
        - normalized_velocity (bool): Whether to include the normalized velocity in the list.
        - scalar_velocity (bool): Whether to include the scalar velocity in the list.

        Returns:
        - List[Union[float, List[float]]]: The list representation of the hand state.
        """
        repr_list = []
        if timestamp:
            repr_list.append(self.last_timestamp)
        if position:
            repr_list += self.position_raw.as_list()
        if normalized_landmarks:
            repr_list += self.normalized_landmarks.flatten().tolist()
        if velocity:
            repr_list += self.velocity_raw.tolist()
        if normalized_landmarks_velocity:
            repr_list += self.normalized_landmarks_velocity.flatten().tolist()
        if filtered_position:
            repr_list += self.position_filtered.as_list()
        if filtered_velocity:
            repr_list += self.velocity_filtered.tolist()
        if filtered_normalized_landmarks:
            repr_list += self.normalized_landmarks_filtered.flatten().tolist()
        if filtered_normalized_landmarks_velocity:
            repr_list += self.normalized_landmarks_velocity_filtered.flatten().tolist()
        if normalized_velocity:
            repr_list += self.normed_velocity.tolist()
        if scalar_velocity:
            repr_list.append(self.scalar_velocity)
        return repr_list

    def __next__(self):
        return self.trajectory.__next__()

    def __getitem__(self, index):
        return self.trajectory[index]
    
    def __str__(self) -> str:
        return f'HandState : {self.position_filtered}'
    
    def __repr__(self) -> str:
        return self.__str__()