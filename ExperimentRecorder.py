#!/usr/bin/env python3

import argparse
import threading
from rgbd.RgbdCameras2 import SimpleRgbdCam as RgbdCamera
import cv2
import numpy as np
import time
import pandas as pd
import os

class ExperimentRecorder:
    def __init__(self, main_path, device_id=None, resolution=(1280, 720), fps=30.0):
        print(f"Recorder created at {main_path}")
        self.main_path = main_path
        self.cam_label = device_id
        self.device_id = device_id
        self.resolution = resolution
        self.fps = fps
        
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.rgbd_camera = RgbdCamera( device_id=device_id,
                                      resolution=resolution,
                                      fps_rgb=fps,      
                                      show_rgb=False,
                                      show_depth=False,)
        self.device_data = self.rgbd_camera.get_device_data()
        res = self.device_data['resolution']
        
        self.path_cam_np = os.path.join(self.main_path, f'cam_{self.cam_label}_{res[0]}_{res[1]}_data.npz')
        if not os.path.exists(self.path_cam_np):
            np.savez(self.path_cam_np, **self.device_data)
            
        self.reset()
        print(f'Recorder with {device_id} built.')
        self.img = None
    
    def reset(self):
        
        self.rgb_frame_series = []
        self.depth_map_series = []
        self.rgb_timestamps_series = []
        self.depth_timestamps_series = []
        
        self.last_rgb_timestamp = 0.
        self.last_depth_timestamp = 0.
        
        self.recording = False

    def init(self):
        self.rgbd_camera.start()
        self.capture_thread = threading.Thread(target=self.capture_task)
        self.record_thread = None
        self.current_path = None
        self.current_recording = None
        self.capture_thread.start()
        self.saving_threads=[]
        print(f'Recorder with {self.device_id} started.')

    def capture_task(self):
        self.new_rec = False
        self.end_rec = False
        
        while self.rgbd_camera.is_on():
            # in case we want to start a new recording
            if self.new_rec:
                self.new_rec = False
                self.recording = True
                
            success, img, map, rgb_timestamp, depth_timestamp = self.rgbd_camera.get_last_frames()
            
            if not success:
                continue
            self.img = img  
            if self.recording:
                is_new_rgb_frame = rgb_timestamp != self.last_rgb_timestamp
                if is_new_rgb_frame:
                    if img is not None and img.size > 0:
                        self.rgb_frame_series.append(img)
                        self.rgb_timestamps_series.append(rgb_timestamp)
                        self.last_rgb_timestamp = rgb_timestamp
                    else:
                        print('Captured empty RGB frame')
                
                is_new_depth_frame = depth_timestamp != self.last_depth_timestamp
                if is_new_depth_frame:
                    if map is not None and map.size > 0:
                        self.depth_map_series.append(map)
                        self.depth_timestamps_series.append(depth_timestamp)
                        self.last_depth_timestamp = depth_timestamp
                    else:
                        print('Captured empty Depth frame')
                
                if not is_new_rgb_frame and not is_new_depth_frame:
                    time.sleep(0.001)
                
                if self.end_rec:
                    self.end_rec = False
                    self.recording = False
    
    def save_data_task(self):
        if self.current_path is None:
            print("No recording to save")
            return
        
        path = self.current_path
        print(f"Start saving {path}")
        path_depth_gzip = self.path_depth_gzip
        path_depth_timestamps_gzip = self.path_depth_timestamps_gzip
        path_depth_timestamps_csv = self.path_depth_timestamps_csv
        path_rgb_timestamps_gzip = self.path_rgb_timestamps_gzip
        path_rgb_timestamps_csv = self.path_rgb_timestamps_csv
        path_vid = self.path_vid
        rgb_frame_series = self.rgb_frame_series
        rgb_timestamps_series = self.rgb_timestamps_series
        depth_map_series = self.depth_map_series
        depth_time_series = self.depth_timestamps_series        
        
        self.ready_to_start_new_rec = True
        
        # # Save video as avi
        # print(f"Saving video at {path_vid}")
        # recorder = cv2.VideoWriter(path_vid, self.fourcc, self.fps, self.device_data['resolution'])
        # print(f'length of rgb_frame_series: {len(rgb_frame_series)}')
        # for frame in rgb_frame_series:
        #     recorder.write(frame)
        #     print(f'writing frame : {frame}')
        # recorder.release()
        # print(f"Finished saving video at {path_vid}")
        
        
        # Save rgb timestamps as gzip and csv
        print(f"Saving rgb timestamps at {path_rgb_timestamps_gzip}")
        zero_start_rgb_timestamps = [t - rgb_timestamps_series[0] for t in rgb_timestamps_series]
        rgb_timestamps_df = pd.DataFrame({'Timestamps': zero_start_rgb_timestamps})
        rgb_timestamps_df.to_pickle(path_rgb_timestamps_gzip, compression='gzip')
        rgb_timestamps_df.to_csv(path_rgb_timestamps_csv, index=False)
        print(f"Finished saving rgb timestamps at {path_rgb_timestamps_gzip}")
        
        # Save depth maps as gzip
        print(f"Saving depth map at {path_depth_gzip}")
        depth_df = pd.DataFrame({'Depth_maps': depth_map_series, 'Date': depth_time_series})
        depth_df['Timestamps'] = depth_df['Date'] - depth_df['Date'][0]
        depth_df.to_pickle(path_depth_gzip, compression='gzip')        
        print(f"Finished saving depth map at {path_depth_gzip}")
        
        # Save depth timestamps as gzip and csv
        print(f"Saving depth timestamps at {path_depth_timestamps_gzip}")
        depth_timestamps_df = pd.DataFrame({'Timestamps': depth_df['Timestamps']})
        depth_timestamps_df.to_pickle(path_depth_timestamps_gzip, compression='gzip')
        depth_timestamps_df.to_csv(path_depth_timestamps_csv, index=False)
        print(f"Finished saving depth timestamps at {path_depth_timestamps_gzip}")
        print(f"Finished saving all files at {path}")


    def write_rgb_frames(self):
        video_writer = cv2.VideoWriter(self.path_vid, self.fourcc, self.fps, self.device_data['resolution'])
        write = True
        frame_count = 0
        while write:
            if len(self.rgb_frame_series) > 0:
                frame = self.rgb_frame_series.pop(0)
                if frame is not None and frame.size > 0:
                    video_writer.write(frame)
                    frame_count += 1
                else:
                    print('Empty frame encountered')
            else:
                time.sleep(0.001)
            if len(self.rgb_frame_series) == 0 and not self.recording:
                write = False
        video_writer.release()
        print(f"Video saved with {frame_count} frames of shape {frame.shape} at {self.path_vid}")

    # def new_record(self, name):
        
    #     print(f"Starting recording {self.device_id} with config {name}")
        
    #     self.current_recording = name
    #     while not self.ready_to_start_new_rec:
    #         time.sleep(0.1)
    #     self.reset()
        
    #     self.current_path= os.path.join(self.main_path, name)
    #     self.path_vid = os.path.join(self.current_path, f'{name}_cam_{self.cam_label}_video.avi')
    #     self.path_depth_gzip = os.path.join(self.current_path,f'{name}_cam_{self.cam_label}_depth_map.gzip')
    #     self.path_depth_timestamps_gzip = os.path.join(self.current_path,f'{name}_cam_{self.cam_label}_depth_timestamps.gzip')
    #     self.path_depth_timestamps_csv = os.path.join(self.current_path,f'{name}_cam_{self.cam_label}_depth_timestamps.csv')
    #     self.path_rgb_timestamps_gzip = os.path.join(self.current_path,f'{name}_cam_{self.cam_label}_rgb_timestamps.gzip')
    #     self.path_rgb_timestamps_csv = os.path.join(self.current_path,f'{name}_cam_{self.cam_label}_rgb_timestamps.csv')
    #     #self.recorder = cv2.VideoWriter(self.path_vid, self.fourcc, 30.0,(1280,720))
    #     self.new_rec = True

    def record_trial(self, trial):
        
        name = trial.label
        print(f"Starting recording {self.device_id} with config {name}")
        
        self.current_recording = name
        self.reset()
        
        self.current_path= os.path.join(self.main_path, name)
        self.path_vid = os.path.join(self.current_path, f'{name}_cam_{self.cam_label}_video.avi')
        self.path_depth_gzip = os.path.join(self.current_path,f'{name}_cam_{self.cam_label}_depth_map.gzip')
        self.path_depth_timestamps_gzip = os.path.join(self.current_path,f'{name}_cam_{self.cam_label}_depth_timestamps.gzip')
        self.path_depth_timestamps_csv = os.path.join(self.current_path,f'{name}_cam_{self.cam_label}_depth_timestamps.csv')
        self.path_rgb_timestamps_gzip = os.path.join(self.current_path,f'{name}_cam_{self.cam_label}_rgb_timestamps.gzip')
        self.path_rgb_timestamps_csv = os.path.join(self.current_path,f'{name}_cam_{self.cam_label}_rgb_timestamps.csv')
        
        self.write_rgb_frames_thread = threading.Thread(target=self.write_rgb_frames)
        self.write_rgb_frames_thread.start()
        
        self.new_rec = True
    
    def stop_record(self):
        self.ready_to_start_new_rec = False
        self.end_rec = True
        if self.current_recording is None:
            print("No recording to stop")
            return
        print(f"Stoping recording {self.device_id} with config {self.current_recording}")
        save_thread = threading.Thread(target=self.save_data_task)
        save_thread.start()
        self.saving_threads.append(save_thread)
        self.write_rgb_frames_thread.join()
    
    def stop(self):
        print(f"Stoping {self.device_id}")
        self.rgbd_camera.stop()
        
        print("Waiting for threads to stop")
        self.capture_thread.join()
        
        print("Capture thread stopped")
        for thread in self.saving_threads:
            thread.join()
            print("Saving thread stopped")
            
        print(f"Stopped {self.device_id}")


        

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
    grasp_int = ExperimentRecorder(**args)
    grasp_int.init()