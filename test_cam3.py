
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
        self.rgbd_camera = RgbdCamera( fps_rgb=self.fps, resolution=self.resolution, show_fps=False, show_stats=True)
        frame_writer_thread = threading.Thread(target=self.write_frames)
        frame_writer_thread.start()
        self.rgbd_camera.start()
        while self.rgbd_camera.is_on():
            t = time.time()
            success, img, map, rgb_timestamp, depth_timestamp = self.rgbd_camera.next_frame()
            if not success:
                # print("Failed to get frame.")
                continue
            # cv2.imshow('img', img)
            # cv2.imshow('map', map)  # Uncomment this line to show the map if needed
            # print(f"RGB timestamp: {rgb_timestamp}, Depth timestamp: {depth_timestamp}")  # Print timestamps for debugging  
            print(f"img shape: {img.shape}, map shape: {map.shape}")
            t_append = time.time()  
            rgb_timestamp_list.append(rgb_timestamp)
            print(f"Append rgb time: {(time.time() - t_append) * 1000} ms")
            t_append = time.time()
            depth_timestamp_list.append(depth_timestamp)
            print(f"Append depth time: {(time.time() - t_append) * 1000} ms")
            t_write = time.time()
            # video_writer.write(img)
            self.video_frame_list.append(img)
            print(f"Write time: {(time.time() - t_write) * 1000} ms")
            
            elapsed_time = time.time() - t
            computer_fps = int(1 / elapsed_time)
            computer_fps_list.append(computer_fps)
            print(f"Computer FPS: {computer_fps}")
        
        
        print(f"Average Computer FPS: {sum(computer_fps_list) / len(computer_fps_list)}")
        print(f"Max Computer FPS: {max(computer_fps_list)}")
        print(f"Min Computer FPS: {min(computer_fps_list)}")
        std_computer_fps = np.std(computer_fps_list)
        print(f"Standard deviation Computer FPS: {std_computer_fps}")
        nb_frames_with_fps_below_desired_minus_std = len([fps for fps in computer_fps_list if fps < 60 - std_computer_fps])
        print(f"Number of frames with FPS below desired FPS minus standard deviation: {nb_frames_with_fps_below_desired_minus_std}")
        frame_writer_thread.join()
        print("End of test.")

if __name__ == "__main__":
    rec = ExperimentRecorder()
    rec.run()