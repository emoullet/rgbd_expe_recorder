import mediapipe as mp
import cv2
import numpy as np
import math
import time
from i_grip.config import _MEDIAPIPE_MODEL_PATH
from typing import List, Dict
# import tensorflow as tf
# print('TENSORFLOW GPU AVAILABLE:')
# print(tf.config.list_physical_devices('GPU'))

class Hands3DDetector:
    """
    Class for detecting and extracting 3D hand landmarks from live stream or video files.
    Args:
        cam_data (dict): The camera data.
        hands (List[str], optional): The list of hands to detect. Defaults to ['left', 'right'].
        running_mode (str, optional): The running mode. Defaults to 'LIVE_STREAM'.
        mediapipe_model_path (str, optional): The path to the mediapipe model. Defaults to _MEDIAPIPE_MODEL_PATH.
        use_gpu (bool, optional): Whether to use GPU for processing. Defaults to True.
    Methods:
        init_landmarker(): Initializes the hand landmarker.
        reset(): Resets the hand landmarker and clears the predictions.
        extract_hands(detection_result, output_image, timestamp_ms): Extracts the hand landmarks from the detection result.
        get_hands_video(frame, depth_frame, timestamp): Processes a video frame and returns the detected hand landmarks.
        get_hands_live_stream(frame, depth_frame): Processes a live stream frame and returns the detected hand landmarks.
        """

    
    LIVE_STREAM_MODE = 'LIVE_STREAM'
    VIDEO_FILE_MODE = 'VIDEO'
    _HANDS_MODE = ['left', 'right']
    
    def __init__(self, cam_data: dict, hands: List[str] = _HANDS_MODE, running_mode: str = LIVE_STREAM_MODE, mediapipe_model_path: str = _MEDIAPIPE_MODEL_PATH, use_gpu: bool = True):
        """
        Initializes the Hands3DDetectors object.
        Parameters:
        - cam_data (dict): The camera data.
        - hands (List[str]): The list of hands to detect. Defaults to _HANDS_MODE.
        - running_mode (str): The running mode. Defaults to LIVE_STREAM_MODE.
        - mediapipe_model_path (str): The path to the mediapipe model. Defaults to _MEDIAPIPE_MODEL_PATH.
        - use_gpu (bool): Whether to use GPU for processing. Defaults to True.
        
        Raises:
        - ValueError: If the hands parameter is invalid.
        - ValueError: If the running_mode parameter is invalid.
        """
        # Set the camera data
        self.cam_data = cam_data
        
        # Validate the hands parameter
        for hand in hands:
            if hand not in self._HANDS_MODE:
                raise ValueError(f'hand must be one of {self._HANDS_MODE}')
        self.hands_to_detect = hands
        self.num_hands = len(hands)

        # Validate the running_mode parameter
        if running_mode not in [self.LIVE_STREAM_MODE, self.VIDEO_FILE_MODE]:
            raise ValueError(f'running_mode must be one of {self.LIVE_STREAM_MODE} or {self.VIDEO_FILE_MODE}')
        
        # Set the base options based on the use_gpu parameter
        if use_gpu:            
            base_options=mp.tasks.BaseOptions(model_asset_path=mediapipe_model_path,
                              delegate = mp.tasks.BaseOptions.Delegate.GPU)
        else:
            base_options=mp.tasks.BaseOptions(model_asset_path=mediapipe_model_path)
            
        # Set the landmarker options based on the running_mode parameter
        if running_mode == self.LIVE_STREAM_MODE:
            self.get_hands = self.get_hands_live_stream
            self.landmarker_options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=self.num_hands,
            min_hand_presence_confidence=0.5,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=self.extract_hands,
            )
        elif running_mode == self.VIDEO_FILE_MODE:
            self.get_hands = self.get_hands_video
            self.landmarker_options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=self.num_hands,
            min_hand_presence_confidence=0.5,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5
            )
            
        # Initialize the hand landmarker and other variables
        self.init_landmarker()
        self.format=mp.ImageFormat.SRGB
        self.stereoInference = StereoInference(self.cam_data)
        
    def init_landmarker(self):
        self.hands_predictions = []
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(self.landmarker_options)
        
    def reset(self):
        self.init_landmarker()
        self.new_frame = False
        
    def extract_hands(self, detection_result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        # Check if there is a detection result
        if detection_result is not None:
            hands_preds = []
            hand_landmarks_list = detection_result.hand_landmarks
            hand_world_landmarks_list = detection_result.hand_world_landmarks
            handedness_list = detection_result.handedness

            # Loop through the detected hands to visualize.
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]
                hand_world_landmarks = hand_world_landmarks_list[idx]
                handedness = handedness_list[idx]
                label = handedness[0].category_name.lower()
                
                # Check if the hand landmarks are valid and if the hand should be detected
                if len(hand_landmarks)>0 and self.depth_map is not None and label in self.hands_to_detect:
                    hand = HandPrediction(handedness, hand_landmarks, hand_world_landmarks, self.depth_map, self.stereoInference)
                    hands_preds.append(hand)
            self.hands_predictions = hands_preds

    def get_hands_video(self, frame, depth_frame, timestamp):
        # Check if the frame and depth frame are valid
        if frame is not None and depth_frame is not None:
            frame_timestamp_ms = round(timestamp*1000)
            mp_image = mp.Image(image_format=self.format, data=frame)
            landmark_results = self.landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            self.depth_map = depth_frame
            self.extract_hands(landmark_results, mp_image, frame_timestamp_ms)
            self.new_frame = False
        return self.hands_predictions
    
    def get_hands_live_stream(self, frame, depth_frame):
        # Check if the frame and depth frame are valid
        if frame is not None and depth_frame is not None:
            frame_timestamp_ms = round(time.time()*1000)
            mp_image = mp.Image(image_format=self.format, data=frame)
            self.depth_map = depth_frame
            self.landmarker.detect_async(mp_image, frame_timestamp_ms)
            self.new_frame = False
        return self.hands_predictions


class HandPrediction:
    """
    Represents a hand prediction object. Embeds the hand landmarks, world landmarks, depth map, and stereo inference.
    Attributes:
        handedness (list): A list of handedness objects.
        landmarks (list): A list of landmark objects.
        world_landmarks (list): A list of world landmark objects.
        depth_map (numpy.ndarray): A numpy array representing the depth map.
        stereo_inference (object): An object representing stereo inference.
    Methods:
        __init__(self, handedness, landmarks, world_landmarks, depth_map, stereo_inference): Initializes a HandPrediction object.
        hand_point(self): Calculates the hand point in 2D and 3D.
        get_landmarks(self): Returns the normalized landmarks.
    """
    def __init__(self, handedness: List, 
                 landmarks: List, 
                 world_landmarks: List, 
                 depth_map: np.ndarray, 
                 stereo_inference: "StereoInference") -> None:
    # def __init__(self, handedness: List[mp.framework.formats.landmark_pb2.NormalizedLandmarkList], 
    #              landmarks: List[mp.framework.formats.landmark_pb2.NormalizedLandmarkList], 
    #              world_landmarks: List[mp.framework.formats.landmark_pb2.NormalizedLandmarkList], 
    #              depth_map: np.ndarray, 
    #              stereo_inference: "StereoInference") -> None:
        self.handedness = handedness
        self.normalized_landmarks = np.array([[l.x,l.y,l.z] for l in landmarks])
        self.world_landmarks = np.array([[l.x,-l.y,l.z] for l in world_landmarks])*1000
        self.label = handedness[0].category_name.lower()
        hand_point2D, hand_point3D = self.hand_point()
        self.position, self.roi = stereo_inference.calc_spatials(hand_point2D, depth_map)
        hand_center = self.position.copy()
        self.world_landmarks = self.world_landmarks + hand_center - hand_point3D
        # self.position = self.position/1000
        
    def hand_point(self):
        """
        Calculates the 2D and 3D coordinates of a hand point.
        Returns:
            tuple: A tuple containing the 2D and 3D coordinates of the hand point.
        """
        hand_point2D = (self.normalized_landmarks[0,:]+self.normalized_landmarks[5,:]+self.normalized_landmarks[17,:])/3
        hand_point3D = (self.world_landmarks[0,:]+self.world_landmarks[5,:]+self.world_landmarks[17,:])/3
        
        # hand_point2D = self.normalized_landmarks[0,:] # wrist
        # hand_point3D = self.world_landmarks[0,:] # wrist
        hand_point2D = (self.normalized_landmarks[0,:]+self.normalized_landmarks[5,:]+self.normalized_landmarks[17,:])/3 # baricenter of wrist, index finger first joint and pinky first joint
        hand_point3D = (self.world_landmarks[0,:]+self.world_landmarks[5,:]+self.world_landmarks[17,:])/3 # baricenter of wrist, index finger first joint and pinky first joint
        return hand_point2D, hand_point3D

    def get_landmarks(self):
        return self.normalized_landmarks
    

class StereoInference:
    """
    Class for performing stereo inference.
    Args:
        cam_data (dict): Camera data containing resolution and hfov.
    Attributes:
        original_width (int): Original width of the camera resolution.
        original_height (int): Original height of the camera resolution.
        hfov (float): Horizontal field of view in radians.
        depth_thres_high (int): High threshold for depth values.
        depth_thres_low (int): Low threshold for depth values.
        box_size (int): Size of the bounding box.
    Methods:
        calc_angle(offset): Calculates the angle based on the offset.
        calc_spatials(normalized_img_point, depth_map, averaging_method): Calculates the spatial coordinates based on the normalized image point and depth map.
    """
    
    def __init__(self, cam_data:Dict) -> None:
        """
        Initializes the StereoInference object.
        Args:
            cam_data (dict): Camera data containing resolution and hfov.
        """

        self.original_width = cam_data['resolution'][0]
        self.original_height = cam_data['resolution'][1]
        
        self.hfov = cam_data['hfov']
        self.hfov = np.deg2rad(self.hfov)

        self.depth_thres_high = 3000
        self.depth_thres_low = 50
        self.box_size = 10


    def calc_angle(self, offset:float) -> float:
        """
        Calculates the angle based on the offset.
        Args:
            offset (float): Offset value.
        Returns:
            float: Calculated angle.
        """
        return math.atan(math.tan(self.hfov / 2.0) * offset / (self.original_width / 2.0))

    def calc_spatials(self, normalized_img_point: tuple, depth_map: np.ndarray, averaging_method: callable = np.mean) -> tuple:
        """
        Calculates the spatial coordinates based on the normalized image point and depth map.
        Args:
            normalized_img_point (tuple): Normalized image point (x, y).
            depth_map (ndarray): Depth map.
            averaging_method (function, optional): Averaging method for calculating average depth. Defaults to np.mean.
        Returns:
            tuple: Spatial coordinates (x, y, z) and bounding box coordinates (xmin, ymin, xmax, ymax).
        """
        if depth_map is None:
            # If no depth map is available, return default values
            print('No depth map available yet')
            return np.array([0,0,0]), None
        
        # Calculate the pixel coordinates based on the normalized image point
        x = normalized_img_point[0]*self.original_width
        y = normalized_img_point[1]*self.original_height
        
        # Calculate the bounding box coordinates
        xmin = max(int(x-self.box_size),0)
        xmax = min(int(x+self.box_size), int(depth_map.shape[1]))
        ymin = max(int(y-self.box_size),0 )
        ymax = min(int(y+self.box_size), int(depth_map.shape[0]))
        
        # Check if the bounding box is flipped
        if xmin > xmax:  
            xmin, xmax = xmax, xmin
        if ymin > ymax:  
            ymin, ymax = ymax, ymin

        # Adjust the bounding box size if it is too small
        if xmin == xmax : 
            xmax = xmin +self.box_size
        if ymin == ymax :
            ymax = ymin +self.box_size

        # Calculate the average depth in the ROI.
        depthROI = depth_map[ymin:ymax, xmin:xmax]
        inThreshRange = (self.depth_thres_low < depthROI) & (depthROI < self.depth_thres_high)
        if depthROI[inThreshRange].any():
            averageDepth = averaging_method(depthROI[inThreshRange])
        else:
            averageDepth = 0

        # Calculate the position and bounding box coordinates in 3D space
        mid_w = int(depth_map.shape[1] / 2) # middle of the depth img
        mid_h = int(depth_map.shape[0] / 2) # middle of the depth img
        bb_x_pos = x - mid_w
        bb_y_pos = y - mid_h
        angle_x = self.calc_angle(bb_x_pos)
        angle_y = self.calc_angle(bb_y_pos)

        z = averageDepth
        x = z * math.tan(angle_x)
        y = -z * math.tan(angle_y)

        return np.array([x,y,z]), (xmin, ymin, xmax, ymax)
        return np.array([x,y,z]), (xmin, ymin, xmax, ymax)
    

