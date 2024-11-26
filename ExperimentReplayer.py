#!/usr/bin/env python3

import numpy as np
import cv2
import pickle 
import os
import pandas as pd
from i_grip import RgbdCameras as rgbd
from i_grip import Hands3DDetectors as hd
from i_grip import Object2DDetectors as o2d
from i_grip import ObjectPoseEstimators as ope
from i_grip import Scene_refactored_multi_thread as sc

#TODO : MODIFY THIS FILE ACCORDING TO YOUR NEEDS

class ExperimentReplayer:
    def __init__(self, device_id, device_data, name = None, display_replay = True, show_depth = False, show_scene = False, save_overlayed_video = True) -> None:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        self.device_id = device_id
        self.show_depth = show_depth
        self.show_scene = show_scene
        self.save_overlayed_video = save_overlayed_video
        
        dataset = "ycbv"        
        self.display_replay = display_replay
        
        self.rgbd_cam = rgbd.RgbdReader()
        # device_data = self.rgbd_cam.get_device_data()
        print(f'cam_data: {device_data}')
        
        hands = ['right', 'left']
        self.hand_detector = hd.Hands3DDetector(device_data, hands = hands, running_mode =
                                            hd.Hands3DDetector.VIDEO_FILE_MODE,
                                            use_gpu=True)
        self.object_detector = o2d.get_object_detector(dataset,
                                                       device_data)
        self.object_pose_estimator = ope.get_pose_estimator(dataset,
                                                            device_data,
                                                            use_tracking = True,
                                                            fuse_detections=False)
        if name is None:
            self.name = f'ExperimentReplayer_{dataset}'
        else:
            self.name = name
            
        self.scene = sc.ReplayScene( device_data, name = f'{self.name}_scene', dataset = dataset)
        
        
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
     
    
    def get_device_id(self):
        return self.device_id
    
    def replay(self, replay, task_hand, task_object, name = None):
        
        self.scene.reset()
        self.rgbd_cam.load_replay(replay)
        self.object_pose_estimator.reset()
        self.hand_detector.reset()
        
        if name is not None:
            cv_window_name = f'{self.name} : Replaying {name}'
            print(f'Replaying {name}')
        else:
            cv_window_name = f'{self.name} : Replaying'
        
        print(f'Task hand: {task_hand}')
        print(f'Task object: {task_object}')
        self.detect = True
        print(f'all timestamps: {self.rgbd_cam.get_timestamps()}')
        
        
        expected_objects = sc.RigidObject.LABEL_EXPE_NAMES
        
        if self.save_overlayed_video :
            saved_imgs = []
        else:
            saved_imgs = None
        
        replay_timestamps = self.rgbd_cam.get_timestamps()
        
        #build a pd dataframe to store the data with the following columns: timestamp, task_hand_found, task_object_found
        # task_hand_found and task_object_found are boolean values
        replay_monitoring = pd.DataFrame(columns = ['timestamp', 'task_hand_found', 'task_object_found'])
        replay_detected_hands = []
        replay_objects_poses = []
        
        for timestamp in replay_timestamps:
            success, img, depth_map = self.rgbd_cam.next_frame()
            width, height = img.shape[1], img.shape[0]
            if height >= width:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                width, height = img.shape[0], img.shape[1]
            if not success:
                continue
            render_img = img.copy()
            to_process_img = img.copy()
            
            to_process_img = cv2.cvtColor(to_process_img, cv2.COLOR_RGB2BGR)
            
            
            # show depth map
            if self.show_depth:
                depthFrameColor = cv2.normalize(depth_map, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depthFrameColor = cv2.equalizeHist(depthFrameColor)
                depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)
                depthFrameColor = cv2.resize(depthFrameColor, (depthFrameColor.shape[1]//2, depthFrameColor.shape[0]//2))
                cv2.imshow(f'depth ', depthFrameColor)
            
            # Hand detection
            detected_hands = self.hand_detector.get_hands(to_process_img, depth_map, timestamp)
            replay_detected_hands.append(detected_hands)
            self.scene.update_hands(detected_hands, timestamp)
            
    
            # Object detection
            if self.detect:
                object_detections, detection_success = self.object_detector.detect_check_and_split(to_process_img, expected_objects.keys(), vertical_split = True)
                print(f'detection_success: {detection_success}')
                self.detect = not detection_success
                    
            else:
                object_detections = None
                
            #check if task hand is detected
            task_hand_detected = False
            for detected_hand in detected_hands:
                if task_hand == detected_hand.label:
                    task_hand_detected = True
                    break
            
            
            #draw bboxes on image
            if object_detections is not None:
                for i in range(len(object_detections)):
                    bbox = object_detections.bboxes[i].cpu().numpy()
                    print(f'bbox draw : {bbox}')
                    # round bbox to int
                    bbox = bbox.astype(int)
                    cv2.rectangle(render_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                    print(f'labels: {object_detections.infos["label"]}')
                    print(f'labels[{i}]: {object_detections.infos["label"][i]}')
                    cv2.putText(render_img, object_detections.infos['label'][i], (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Object pose estimation
            objects_poses = self.object_pose_estimator.estimate(to_process_img, detections = object_detections)
            
            # filter out objects that are not in the expected objects
            label_to_del = []
            if objects_poses is not None:
                for object_pose_label in objects_poses.keys():
                    if object_pose_label not in expected_objects:
                        label_to_del.append(object_pose_label)
            for object_pose_label in label_to_del:
                del objects_poses[object_pose_label]
                    
            #check if task object is detected
            task_object_detected = False
            if objects_poses is not None:
                for object_pose_label in objects_poses.keys():
                    if task_object == expected_objects[object_pose_label]:
                        task_object_detected = True
                        break
            replay_objects_poses.append(objects_poses)
                    
            self.scene.update_objects(objects_poses, timestamp)
            
            # store the task verification in the replay_monitoring dataframe
            replay_monitoring.loc[len(replay_monitoring)] = [timestamp, task_hand_detected, task_object_detected]
                
            if self.display_replay or self.save_overlayed_video :
                self.scene.render(render_img)                
                if self.display_replay:
                    cv2.imshow(cv_window_name, render_img)
                if self.save_overlayed_video:
                    save_img = cv2.resize(render_img, (height//2, width//2))
                    saved_imgs.append(save_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('end')
                self.stop()
                break
        self.scene.pause_scene_display()
        hands_data = self.scene.get_hands_data()
        objects_data = self.scene.get_objects_data()
        replay_data_dict = {'timestamps': replay_timestamps, 'hands': replay_detected_hands, 'objects': replay_objects_poses}
        
        
        return hands_data, objects_data, replay_monitoring, saved_imgs, replay_data_dict
        
    def stop(self):
        print("Stopping experiment replayer...")
        print("Stopping object estimator...")
        self.object_pose_estimator.stop()
        print("Stopped object estimator...")
        print("Stopping scene...")
        self.scene.stop()
        print("Stopped scene...")
        
        cv2.destroyAllWindows()
        

        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-hd', '--hand_detection', choices=['mediapipe', 'depthai', 'hybridOAKMediapipe'],
                        default = 'hybridOAKMediapipe', help="Hand pose reconstruction solution")
    parser.add_argument('-od', '--object_detection', choices=['cosypose, megapose'],
                        default = 'cosypose', help="Object pose reconstruction detection")
    args = vars(parser.parse_args())

    # if args.hand_detection == 'mediapipe':
    #     import mediapipe as mp
    # else:
    #     import depthai as dai
    
    # if args.object_detection == 'cosypose':
    #     import cosypose
    i_grip = ExperimentReplayer(**args)
    i_grip.run()