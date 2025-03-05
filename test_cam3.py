
from RgbdCameras2 import SimpleRgbdCam as RgbdCamera
import time
import numpy as np  
import cv2
import threading

# COUCOU SARA
    
class ExperimentRecorder:
    def __init__(self):
        self.video_frame_list = []
        self.fps = 60.0
        self.resolution = RgbdCamera._720P
        
    def write_frames( self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter('output.avi', fourcc, self.fps, self.resolution)
        write = True
        while write:
            if len(self.video_frame_list) > 0:
                frame = self.video_frame_list.pop(0)
                video_writer.write(frame)
            else:
                time.sleep(0.001)
            if len(self.video_frame_list) == 0 and not self.rgbd_camera.is_on():
                write = False
        video_writer.release()
        print("Video saved.")
    
    def run(self):
        
        rgb_timestamp_list = []
        depth_timestamp_list = []
        computer_fps_list = []
        last_rgb_timestamp = 0.
        last_depth_timestamp = 0.
        
        
        self.rgbd_camera = RgbdCamera( fps_rgb=self.fps, resolution=self.resolution, show_fps=False, show_stats=True)
        frame_writer_thread = threading.Thread(target=self.write_frames)
        
        frame_writer_thread.start()
        self.rgbd_camera.start()
        
        while self.rgbd_camera.is_on():
            t = time.time()
            success, img, map, rgb_timestamp, depth_timestamp = self.rgbd_camera.get_frame()
            if not success:
                # print("Failed to get frame.")
                continue
            
            is_new_rgb_frame = rgb_timestamp != last_rgb_timestamp
            if is_new_rgb_frame:
                rgb_timestamp_list.append(rgb_timestamp)
                self.video_frame_list.append(img)
                last_rgb_timestamp = rgb_timestamp
                print("New RGB frame.")
                print(f'RGB frame shape: {img.shape}')
            
            is_new_depth_frame = depth_timestamp != last_depth_timestamp
            if is_new_depth_frame:
                depth_timestamp_list.append(depth_timestamp)
                last_depth_timestamp = depth_timestamp
            
                print("New Depth frame.")

            if not is_new_rgb_frame and not is_new_depth_frame:
                time.sleep(0.001)
                print("No new frame.")
                continue
            
            elapsed_time = time.time() - t
            computer_fps = int(1 / elapsed_time)
            computer_fps_list.append(computer_fps)
        
        
        std_computer_fps = np.std(computer_fps_list)
        nb_frames_with_fps_below_desired_minus_std = len([fps for fps in computer_fps_list if fps < 60 - std_computer_fps])
        
        print(f"Average Computer FPS: {sum(computer_fps_list) / len(computer_fps_list)}")
        print(f"Max Computer FPS: {max(computer_fps_list)}")
        print(f"Min Computer FPS: {min(computer_fps_list)}")
        print(f"Standard deviation Computer FPS: {std_computer_fps}")
        print(f"Number of frames with FPS below desired FPS minus standard deviation: {nb_frames_with_fps_below_desired_minus_std}")
        
        frame_writer_thread.join()
        print("End of test.")

if __name__ == "__main__":
    rec = ExperimentRecorder()
    rec.run()