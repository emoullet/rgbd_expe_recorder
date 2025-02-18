import depthai as dai
import numpy as np
import cv2
import time
import threading
from datetime import timedelta

class rgb_depth:
    def __init__(self):
    
        self.pipeline = dai.Pipeline()
        
        self.pipeline.setXLinkChunkSize(0)
        self.queue_size = 2
        self.blocking_queue = True
        self.fps = 60
        self.fps_depth = 60

        monoLeft = self.pipeline.create(dai.node.MonoCamera)
        monoRight = self.pipeline.create(dai.node.MonoCamera)
        
        monores = dai.MonoCameraProperties.SensorResolution.THE_400_P
        monoLeft.setResolution(monores)
        monoLeft.setCamera("left")
        monoRight.setResolution(monores)
        monoRight.setCamera("right")
        
        
        monoLeft.setFps(self.fps_depth)
        monoRight.setFps(self.fps_depth)
        
        # color = self.pipeline.create(dai.node.ColorCamera)
        color = self.pipeline.createColorCamera()
        color.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
        color.setCamera("color")
        color.setIspScale(2, 3)
        color.setInterleaved(True)
        # color.setVideoSize(1280, 720)
        color.setFps(self.fps)
        
        monoLeft.setNumFramesPool(2)
        monoRight.setNumFramesPool(2)
        color.setNumFramesPool(2,2,2,2,2)
        
        
        self.stereo = self.pipeline.create(dai.node.StereoDepth)
        self.stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
        self.stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        self.stereo.setLeftRightCheck(True)
        self.stereo.setExtendedDisparity(False)
        # stereo.setSubpixel(True)
    
        self.stereo.left.setBlocking(self.blocking_queue)
        self.stereo.left.setQueueSize(self.queue_size)
        self.stereo.right.setBlocking(self.blocking_queue)
        self.stereo.right.setQueueSize(self.queue_size)
        
        

        color_out = self.pipeline.create(dai.node.XLinkOut)
        color_out.setStreamName("video")
        color_out.input.setBlocking(self.blocking_queue)
        color_out.input.setQueueSize(self.queue_size)
        
        depth_out = self.pipeline.create(dai.node.XLinkOut)
        depth_out.setStreamName("disparity")
        depth_out.input.setBlocking(self.blocking_queue)
        depth_out.input.setQueueSize(self.queue_size)

        color.isp.link(color_out.input)
        monoLeft.out.link(self.stereo.left)
        monoRight.out.link(self.stereo.right)

        self.stereo.disparity.link(depth_out.input)

        disparityMultiplier = 255.0 / self.stereo.initialConfig.getMaxDisparity()
    
    def run(self):
        with dai.Device(self.pipeline, maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS) as self.device:
            queue = self.device.getOutputQueue("video", self.queue_size, self.blocking_queue)   
            old_rgb_timestamp = 1
            self.current_depth_timestamp = 1
            self.running = True
            self.depth_frame = None
            self.depth_fps = 0
            
            depth_thread = threading.Thread(target=self.depth_collection_thread)   
            depth_thread.start()
            
            rgb_fps_list = []
            depth_fps_list = []
            rgb_to_depth_latency_list = []
            
            while self.running:
                color_msg = queue.get()
                color_frame = color_msg.getCvFrame()
                current_rgb_timestamp = color_msg.getTimestamp().total_seconds()
                
                cv2.imshow("color", color_frame)
                if self.depth_frame is not None:
                    cv2.imshow("depth", self.depth_frame)
                
                # if old_rgb_timestamp:
                #     print(f"RGB FPS: {int(1.0 / (current_rgb_timestamp - old_rgb_timestamp))}")
                rgb_to_depth_latency = current_rgb_timestamp - self.current_depth_timestamp
                # print(f"RGB-Depth latency: {rgb_to_depth_latency*1000:.1f} ms")
                

                rgb_fps_list.append(int(1.0 / (current_rgb_timestamp - old_rgb_timestamp)))
                depth_fps_list.append(self.depth_fps)
                rgb_to_depth_latency_list.append(rgb_to_depth_latency * 1000)  # storing latency in ms
                
                old_rgb_timestamp = current_rgb_timestamp
                
                if cv2.waitKey(1) == ord("q"):
                    self.running = False
                    break
            

            depth_thread.join()
            print(f"Average RGB FPS: {sum(rgb_fps_list) / len(rgb_fps_list)}")           
            print(f"Max RGB FPS: {max(rgb_fps_list)}")  
            print(f"Min RGB FPS: {min(rgb_fps_list)}")
            std_rgb_fps = np.std(rgb_fps_list)  
            print(f"Standard deviation RGB FPS: {std_rgb_fps}")
            nb_frames_with_fps_below_desired_minus_std = len([fps for fps in rgb_fps_list if fps < self.fps - std_rgb_fps])
            print(f"Number of frames with FPS below desired FPS minus standard deviation: {nb_frames_with_fps_below_desired_minus_std}")
             
            print(f"Average Depth FPS: {sum(depth_fps_list) / len(depth_fps_list)}")   
            print(f"Max Depth FPS: {max(depth_fps_list)}")
            print(f"Min Depth FPS: {min(depth_fps_list)}")            
            std_depth_fps = np.std(depth_fps_list)
            print(f"Standard deviation Depth FPS: {std_depth_fps}")
            nb_depth_frames_below_threshold = len([fps for fps in depth_fps_list if fps < self.fps_depth - std_depth_fps])
            print(f"Number of Depth frames below threshold minus standard deviation: {nb_depth_frames_below_threshold}")
             
            rgb_to_depth_latency_list= rgb_to_depth_latency_list[10:]                   
            print(f"Average RGB-Depth latency: {sum(rgb_to_depth_latency_list) / len(rgb_to_depth_latency_list)} ms")
            print(f"Max RGB-Depth latency: {max(rgb_to_depth_latency_list)} ms")
            print(f"Min RGB-Depth latency: {min(rgb_to_depth_latency_list)} ms")
            
    def depth_collection_thread(self):
        self.old_depth_timestamp = 1
        self.disparity_queue = self.device.getOutputQueue("disparity", self.queue_size, self.blocking_queue)
        while self.running:
            depth_msg = self.disparity_queue.get()
            self.depth_frame = depth_msg.getCvFrame()
            self.current_depth_timestamp = depth_msg.getTimestamp().total_seconds()  
            if self.old_depth_timestamp:
                self.depth_fps = int(1.0 / (self.current_depth_timestamp - self.old_depth_timestamp))
                # print(f"Depth FPS: {self.depth_fps}")
            self.old_depth_timestamp = self.current_depth_timestamp
            # cv2.imshow("depth", depth_frame)

if __name__ == "__main__":
    rd = rgb_depth()
    rd.run()
    cv2.destroyAllWindows()