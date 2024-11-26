#!/usr/bin/env python3

import argparse
from i_grip import Scene_refactored_multi_thread as sc
import cv2
import pandas as pd
from typing import Any, List, Optional, Dict

#TODO : MODIFY THIS FILE ACCORDING TO YOUR NEEDS

class ExperimentAnalyser:
    """
    A class to analyze experimental data.

    Attributes:
        data (Any): The experimental data to be analyzed.
        results (Dict): The results of the analysis.
        config (Dict): Configuration settings for the analysis.
        logger (logging.Logger): Logger for the class.

    Methods:
        __init__(self, data: Any, config: Optional[Dict] = None) -> None:
            Initialize the ExperimentAnalyser with data and optional configuration.
        preprocess_data(self) -> None:
            Preprocess the experimental data before analysis.
        analyze(self) -> None:
            Perform the analysis on the experimental data.
        postprocess_results(self) -> None:
            Postprocess the results of the analysis.
        save_results(self, filepath: str) -> None:
            Save the analysis results to a file.
        load_results(self, filepath: str) -> None:
            Load analysis results from a file.
        get_summary(self) -> Dict:
            Get a summary of the analysis results.
        log_info(self, message: str) -> None:
            Log an informational message.
        log_error(self, message: str) -> None:
            Log an error message.
    """
    
    def __init__(self, device_id: int, device_data: Dict, name: Optional[str] = None, show_video: bool = True, save_scene: bool = True) -> None:
        """
        Initializes an instance of ExperimentAnalyser.
        Parameters:
        - device_id (int): The ID of the device.
        - device_data (Dict): The data of the device.
        - name (Optional[str]): The name of the ExperimentAnalyser instance. If not provided, it will be set to 'ExperimentAnalyser'.
        - show_video (bool): Whether to show the video or not. Default is True.
        - save_scene (bool): Whether to save the scene or not. Default is True.
        """

        
        self.show_video = show_video
        self.save_scene = save_scene
        self.device_data = device_data
        if name is None:
            self.name = f'ExperimentAnalyser'
        else:
            self.name = name
        # self.scene = sc.AnalysisScene( device_data, dataset='ycbv', name = f'{self.name} scene', is_displayed=self.show_scene)
        self.device_id = device_id
    
    def get_device_id(self):
        return self.device_id
    
    def analyse(self, task_hand: str, task_object: str, task_grip: str, timestamps: List[float], hands: List[Any], objects: List[Any], video_path: str, name: Optional[str] = None, save_scene_path: Optional[str] = None) -> pd.DataFrame:
        if save_scene_path is not None and self.save_scene:
            draw_mesh = True
        else:
            draw_mesh = False
        self.scene= sc.AnalysisScene( self.device_data, dataset='ycbv', name = f'{self.name} scene', is_displayed=self.save_scene, draw_mesh = draw_mesh)
        # self.scene.reset()
        # while not self.scene.is_ready():
        #     cv2.waitKey(10)
        #     print('waiting for scene')
        if name is not None:
            cv_window_name = f'{self.name} : Replaying {name}'
        else:
            cv_window_name = f'{self.name} : Replaying'
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f'Error opening video file {video_path}')
            return None
        
        
        expected_objects = sc.RigidObject.LABEL_EXPE_NAMES
        
        # create a pd dataframe to store the target data with the following columns: timestamp, estimated_target_object, estimated_target_grip, estimated_target_time_to_impact, task_object_found, task_grip_found, task_time_to_impact
        # analysis_data = pd.DataFrame(columns=['timestamp', 'task_hand_info_found', 'estimated_target_object', 'estimated_target_grip', 'estimated_target_time_to_impact', 'task_target_found', 'task_grip_found', 'diff_task_time_to_impact', 'decisive_metric', 'metric_confidence', 'min_distance_to_target', 'min_distance_between_targets', 'max_distance_derivative', 'hand_velocity', 'hand_pos_x', 'hand_pos_y', 'hand_pos_z'])
        analysis_data = pd.DataFrame()
        if len(objects) >= 6:
            obj_init = objects[10]
        else:
            obj_init = objects[-1]
        for i, timestamp in enumerate(timestamps):
            
            self.scene.update_hands(hands[i], timestamp=timestamp)
            print(f'hands[i]: {hands[i]}')
            if i >= 5:
                obj = objects[i]
            else:
                obj = obj_init
            self.scene.update_objects(obj, timestamp=timestamp, propagate=True)
            self.scene.propagate_hands(timestamp=timestamp)
            # self.scene.propagate_objects(timestamp=timestamp)
            # self.scene.set_not_updated()
            # cv2.waitKey(100)
            if not draw_mesh:
                self.scene.update_meshes(timestamp=timestamp)
                self.scene.detect_intentions(timestamp=timestamp)
            # self.scene.propagate(timestamp=timestamp)
            
            if self.show_video:
                success, img = cap.read()
                if not success:
                    print(f'Error reading frame {i} from video file {video_path}')
                    continue
                #write the frame number on the frame
                cv2.putText(img, f'frame {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow(cv_window_name, img)
                cv2.waitKey(1)
                
            targets_info= self.scene.get_targets_info()
            print(f'at timestamp {timestamp}, targets_info: {targets_info}')
            
            if save_scene_path is not None and self.save_scene:
                print('trying to save scene')
                self.scene.save_scene(f'{save_scene_path}_{i}.png')
                # cv2.waitKey(1000)
                cv2.imwrite(f'{save_scene_path}_im_{i}.png', img)
            
            task_hand_info_found = task_hand  in targets_info
            
            if not task_hand_info_found:
                print(f'Hand {task_hand} not found in the scene at timestamp {timestamp}')
                estimated_target_grip = None
                estimated_target_object = None
                estimated_target_time_to_impact = None
                task_object_found = False
                task_grip_found = False
                diff_task_time_to_impact = None
                decisive_metric = None
                metric_confidence = None
                min_distance_to_target = None
                min_distance_between_targets = None
                max_distance_derivative = None
                hand_velocity = None
            else:
                task_hand_target_info = targets_info[task_hand]
            
                estimated_target_object = task_hand_target_info['object']
                estimated_target_grip = task_hand_target_info['grip']
                estimated_target_time_to_impact = task_hand_target_info['time_to_impact']
                # decisive_metric = task_hand_target_info['decisive_metric']
                # metric_confidence = task_hand_target_info['metric_confidence']
                # min_distance_to_target = task_hand_target_info['min_distance_to_target']
                # min_distance_between_targets = task_hand_target_info['min_distance_between_targets']
                # max_distance_derivative = task_hand_target_info['max_distance_derivative']
                # hand_velocity = task_hand_target_info['hand_velocity']
                # hand_pos_x = task_hand_target_info['hand_pos_x']
                # hand_pos_y = task_hand_target_info['hand_pos_y']
                # hand_pos_z = task_hand_target_info['hand_pos_z']
                
                task_object_found = estimated_target_object == task_object
                task_grip_found = estimated_target_grip == task_grip
                if timestamps[-1] is None or task_hand_target_info['time_to_impact'] is None:
                    time_to_impact = None
                    # diff_task_time_to_impact = None
                else:
                    time_to_impact = timestamps[-1] - timestamp
                    # diff_task_time_to_impact = task_hand_target_info['time_to_impact'] - time_to_impact
                    
            analysis_data.loc[i, 'timestamp'] = timestamp
            analysis_data.loc[i, 'task_hand_info_found'] = task_hand_info_found
            analysis_data.loc[i, 'task_target_found'] = task_object_found
            analysis_data.loc[i, 'task_grip_found'] = task_grip_found
            analysis_data.loc[i, 'estimated_target_time_to_impact'] = estimated_target_time_to_impact
            
            if task_hand_info_found:
                for c_label in task_hand_target_info.keys():
                    analysis_data.loc[i, c_label] = task_hand_target_info[c_label]
            
            # analysis_data.loc[i] = [timestamp, task_hand_info_found, estimated_target_object, estimated_target_grip, estimated_target_time_to_impact, task_object_found, task_grip_found, diff_task_time_to_impact, decisive_metric, metric_confidence, min_distance_to_target, min_distance_between_targets, max_distance_derivative, hand_velocity, hand_pos_x, hand_pos_y, hand_pos_z]
            
            
            
        
        self.scene.pause_scene_display()
        return analysis_data
        
    def stop(self):
        if hasattr(self, 'scene'):
            print("Stopping scene...")
            self.scene.stop()
        print("Stopped scene...")
        
        cv2.destroyAllWindows()
        

        
