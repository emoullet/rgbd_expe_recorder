import depthai as dai
import cv2
import numpy as np
import time
from typing import Optional, Any, Dict, List
from datetime import timedelta
import threading


class SimpleRgbdCam:
    _720P = (1280, 720)
    _1080P = (1920., 1080.)
    _480P = (640., 480.)
    _RGB_MODE = 'RGB'
    _BGR_MODE = 'BGR'
    def __init__(self,
                 device_id=None,
                 fps_rgb=60,
                 fps_depth=None,
                 resolution=_720P,
                 show_rgb=True,
                 show_depth=True,
                 show_stats=False,
                 show_fps=False,
                 color_mode='RGB'
                 ):
        self.device_id = device_id
        self.running = False
        self.fps_rgb = fps_rgb
        self.fps_depth = fps_depth if fps_depth is not None else fps_rgb
        self.resolution = resolution
        self.show_rgb = show_rgb
        self.show_depth = show_depth
        self.show_stats = show_stats
        self.show_fps = show_fps
        if color_mode not in [self._RGB_MODE, self._BGR_MODE]:
            raise ValueError(f'color_mode must be one of {self._RGB_MODE} or {self._BGR_MODE}')
        else:
            self.color_mode = color_mode
        
        self.build_device()
        
        self.cam_data = {}
        self.cam_data['resolution'] = (int(self.resolution[0]), int(self.resolution[1]))
        calibData = self.device.readCalibration()
        self.cam_data['matrix'] = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB, self.cam_data['resolution'][0], self.cam_data['resolution'][1]))
        self.cam_data['hfov'] = calibData.getFov(dai.CameraBoardSocket.RGB)
        
    def build_device(self):
        
        if self.device_id is None:  
            self.device = dai.Device()
        else:
            self.device = dai.Device(dai.DeviceInfo(self.device_id), maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS)
    
        self.pipeline = dai.Pipeline()
        
        self.pipeline.setXLinkChunkSize(0)
        self.queue_size = 2
        self.blocking_queue = True

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
        color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        color.setCamera("color")
        if self.resolution == self._720P:
            color.setIspScale(2, 3)
        elif self.resolution == self._480P:
            color.setIspScale(1, 3)
            
        color.setInterleaved(True)
        color.setFps(self.fps_rgb)
        
        if self.color_mode == self._RGB_MODE:
            color.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
            print("Setting RGB color mode") 
        elif self.color_mode == self._BGR_MODE:
            color.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            print("Setting BGR color mode")
        
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
        
        # disparity_out = self.pipeline.create(dai.node.XLinkOut)
        # disparity_out.setStreamName("disparity")
        # disparity_out.input.setBlocking(self.blocking_queue)
        # disparity_out.input.setQueueSize(self.queue_size)
        
        depth_out = self.pipeline.create(dai.node.XLinkOut)
        depth_out.setStreamName("depth")
        depth_out.input.setBlocking(self.blocking_queue)
        depth_out.input.setQueueSize(self.queue_size)
        
        color.isp.link(color_out.input)
        monoLeft.out.link(self.stereo.left)
        monoRight.out.link(self.stereo.right)

        self.stereo.depth.link(depth_out.input)
        # self.stereo.disparity.link(disparity_out.input)

        
        self.device.startPipeline(self.pipeline)
        
    
    def run(self):
        self.rgb_frame = None
        self.depth_frame = None
        
        self.running = True
        self.current_depth_fps = 0
        
        
        self.rgb_fps_list = []
        self.depth_fps_list = []
        self.rgb_to_depth_latency_list = []
        
        self.current_rgb_timestamp = 0
        self.current_depth_timestamp = 0
        self.rgb_thread = threading.Thread(target=self.rgb_collection_thread)
        self.depth_thread = threading.Thread(target=self.depth_collection_thread)   
        self.rgb_thread.start()
        self.depth_thread.start()
    
    def rgb_collection_thread(self):
        print("Starting RGB collection thread...")
        old_rgb_timestamp = 1 
        queue = self.device.getOutputQueue("video", self.queue_size, self.blocking_queue)  
        while self.running:
            color_msg = queue.get()
            self.rgb_frame = color_msg.getCvFrame()
            self.current_rgb_timestamp = color_msg.getTimestamp().total_seconds()
            
            
            
            if self.show_fps or self.show_stats:
                current_rgb_fps = int(1.0 / (self.current_rgb_timestamp - old_rgb_timestamp))
                
            if self.show_fps:
                print(f"RGB FPS: {current_rgb_fps}")
            
            if self.show_stats:
                self.rgb_fps_list.append(current_rgb_fps)
                
            old_rgb_timestamp = self.current_rgb_timestamp
            
        
        self.depth_thread.join()
        
    def depth_collection_thread(self):
        print("Starting Depth collection thread...")
        self.current_depth_timestamp = 0
        self.old_depth_timestamp = 1
        self.depth_queue = self.device.getOutputQueue("depth", self.queue_size, self.blocking_queue)
        
        while self.running:
            depth_msg = self.depth_queue.get()
            self.depth_frame = depth_msg.getCvFrame()
            self.current_depth_timestamp = depth_msg.getTimestamp().total_seconds()  
            
            if self.show_fps or self.show_stats:
                self.current_depth_fps = int(1.0 / (self.current_depth_timestamp - self.old_depth_timestamp))
                
            if self.show_fps:
                print(f"Depth FPS: {self.current_depth_fps}")
                
            if self.show_stats:
                self.depth_fps_list.append(self.current_depth_fps)
            self.old_depth_timestamp = self.current_depth_timestamp
            # cv2.imshow("depth", depth_frame)
    
    def next_frame(self):
        
        success = True
        if self.rgb_frame is None or self.depth_frame is None or self.current_rgb_timestamp == 0 or self.current_depth_timestamp == 0:
            success = False
            return success, self.rgb_frame, self.depth_frame, self.current_rgb_timestamp, self.current_depth_timestamp
        
        if self.show_rgb or self.show_depth:
            self.show_frames()
            
        if self.show_fps or self.show_stats:
            rgb_to_depth_latency = self.current_rgb_timestamp - self.current_depth_timestamp
        
        if self.show_fps:
            print(f"RGB-Depth latency: {rgb_to_depth_latency*1000:.1f} ms")
            
        if self.show_stats:
            self.rgb_to_depth_latency_list.append(rgb_to_depth_latency * 1000)  # storing latency in ms
            
        return success, self.rgb_frame, self.depth_frame, self.current_rgb_timestamp, self.current_depth_timestamp
    
    def show_frames(self):
        if self.show_rgb:
            if self.rgb_frame is not None:
                cv2.imshow("color", self.rgb_frame)
            
        if self.show_depth:
            if self.depth_frame is not None:
                depthFrameColor = cv2.normalize(self.depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depthFrameColor = cv2.equalizeHist(depthFrameColor)
                depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)
                cv2.imshow("depth", depthFrameColor)
            
        if cv2.waitKey(1) == ord("q"):
            self.stop()
            
    def print_stats(self):
        print(f"Average RGB FPS: {sum(self.rgb_fps_list) / len(self.rgb_fps_list)}")           
        print(f"Max RGB FPS: {max(self.rgb_fps_list)}")  
        print(f"Min RGB FPS: {min(self.rgb_fps_list)}")
        std_rgb_fps = np.std(self.rgb_fps_list)  
        print(f"Standard deviation RGB FPS: {std_rgb_fps}")
        nb_frames_with_fps_below_desired_minus_std = len([fps for fps in self.rgb_fps_list if fps < self.fps_rgb - std_rgb_fps])
        print(f"Number of frames with FPS below desired FPS minus standard deviation: {nb_frames_with_fps_below_desired_minus_std}")
            
        print(f"Average Depth FPS: {sum(self.depth_fps_list) / len(self.depth_fps_list)}")   
        print(f"Max Depth FPS: {max(self.depth_fps_list)}")
        print(f"Min Depth FPS: {min(self.depth_fps_list)}")            
        std_depth_fps = np.std(self.depth_fps_list)
        print(f"Standard deviation Depth FPS: {std_depth_fps}")
        nb_depth_frames_below_threshold = len([fps for fps in self.depth_fps_list if fps < self.fps_depth - std_depth_fps])
        print(f"Number of Depth frames below threshold minus standard deviation: {nb_depth_frames_below_threshold}")
            
        self.rgb_to_depth_latency_list= self.rgb_to_depth_latency_list[10:]                   
        print(f"Average RGB-Depth latency: {sum(self.rgb_to_depth_latency_list) / len(self.rgb_to_depth_latency_list)} ms")
        print(f"Max RGB-Depth latency: {max(self.rgb_to_depth_latency_list)} ms")
        print(f"Min RGB-Depth latency: {min(self.rgb_to_depth_latency_list)} ms")
            

    def start(self):
        self.run()
        
    def stop(self):
        self.running = False
        self.device.close()
        cv2.destroyAllWindows()
        if self.show_stats:
            self.print_stats()
        
    def is_on(self):
        return self.running
    
    def get_device_data(self):
        return {'resolution': (1280, 720)}

class RgbdCamera:
    """
    A class to represent an RGB-D camera.

    Attributes:
        _720P (List[float]): Resolution for 720P.
        _1080P (List[float]): Resolution for 1080P.
        _480P (List[float]): Resolution for 48supportedTypes: [MONO], hasAutofocus: 0, hasAutofocusIC: 0, name: right}
        _RGB_MODE (str): RGB color mode.
        _BGR_MODE (str): BGR color mode.
        cam_auto_mode (bool): Flag for automatic camera mode.
        device_id (Optional[str]): Device ID.
        replay (bool): Flag indicating whether to replay data.
        fps (float): Frames per second.
        resolution (List[float]): Resolution of the camera.
        print_rgb_stereo_latency (bool): Flag indicating whether to print RGB stereo latency.
        show_disparity (bool): Flag indicating whether to show disparity.
        color_mode (str): Color mode.
        cam_data (Dict): Camera data.
        frame (Optional[np.ndarray]): Current frame.
        new_frame (bool): Flag indicating whether a new frame is available.
        device (dai.Device): DepthAI device.
        lensPos (int): Lens position.
        expTime (int): Exposure time.
        sensIso (int): ISO sensitivity.
        wbManual (int): Manual white balance.
        rgb_res (dai.ColorCameraProperties.SensorResolution): RGB camera resolution.
        mono_res (dai.MonoCameraProperties.SensorResolution): Mono camera resolution.
        rgbQ (dai.DataOutputQueue): RGB output queue.
        depthQ (dai.DataOutputQueue): Depth output queue.
        cam_out (dai.node.XLinkOut): XLinkOut node for RGB camera.
        depth_out (dai.node.XLinkOut): XLinkOut node for depth map.
        depth_map (Optional[np.ndarray]): Current depth map.
        timestamp (float): Timestamp of the current frame.
        video (cv2.VideoCapture): Video capture object for replay.
        nb_frames (int): Number of frames in the video replay.
        current_frame_index (int): Current frame index in the video replay.
        timestamps (List[float]): Timestamps of the frames in the video replay.
        depth_maps (List[np.ndarray]): Depth maps of the frames in the video replay.
        on (bool): Flag indicating whether the camera is on.

    Methods:
        __init__(self, replay: bool = False, replay_data: Optional[Any] = None, cam_params: Optional[Dict] = None, device_id: Optional[str] = None, fps: float = 30., resolution: List[float] = _480P, print_rgb_stereo_latency: bool = False, show_disparity: bool = False, color_mode: str = _BGR_MODE) -> None:
            Instantiate a RGB-D Camera object.
        build_device(self) -> None:
            Build the device and initialize camera settings.
        create_RGB_pipeline(self) -> None:
            Create a pipeline for the RGB camera.
        create_stereo_pipeline(self) -> None:
            Create a pipeline for the stereo cameras.
        create_rgb_only_pipeline(self) -> None:
            Create a pipeline for RGB-D cameras, but only RGB frames are captured.
        create_rgb_depth_pipeline(self) -> None:
            Create a pipeline for RGB-D cameras, captures RGB and depth frames asynchronously.
        create_rgb_depth_synced_pipeline(self, max_rgb_depth_latency) -> None:
            Create a pipeline for RGB-D cameras, captures RGB and depth frames synchronously, within a given latency tolerance.
        load_replay(self, replay: dict) -> None:
            Load a replay file.
        next_frame_livestream(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
            Retrieve the next frame from the livestream.
        next_frame_video(self) -> Tuple[bool, np.ndarray, np.ndarray]:
            Read the next frame from the video and update the current frame, depth map, and timestamp.
        get_depth_map(self) -> np.ndarray:
            Get the current depth map.
        start(self) -> None:
            Start the RGB-D camera.
        stop(self) -> None:
            Stop the RGB-D camera.
        get_res(self) -> Tuple[int, int]:
            Get the resolution of the RGB-D camera.
        get_device_data(self) -> Dict:
            Get the device data of the RGB-D camera.
        get_num_frames(self) -> int:
            Get the total number of frames in the video replay.
        get_timestamps(self) -> List[float]:
            Get the timestamps of the frames in the video replay.
        is_on(self) -> bool:
            Check if the RGB-D camera is currently on.
        is_on_replay(self) -> bool:
            Check if the RGB-D camera is currently on replay mode.
    """
    _720P = [1280., 720.]
    _1080P = [1920., 1080.]
    _480P = [640., 480.]
    _480P = [640., 360.]
    _RGB_MODE = 'RGB'
    _BGR_MODE = 'BGR'
    
    def __init__(self, replay: bool = False, 
                 replay_data: Optional[Any] = None, 
                 cam_params: Optional[Dict] = None, 
                 device_id: Optional[str] = None, 
                 fps: float = 30., 
                 resolution: List[float] = _480P, 
                 print_rgb_stereo_latency: bool = False, 
                 show_disparity: bool = False,
                 color_mode: str = _BGR_MODE,
                 auto_focus = True,
                 get_depth = True,
                 sync_depth = True) -> None:
        
        """Instantiate a RGBd Camera object.
            Args:
                replay (bool, optional): Flag indicating whether to replay data. Defaults to False.
                replay_data (Any, optional): Data for replaying. Defaults to None.
                cam_params (Dict, optional): Camera parameters. Defaults to None.
                device_id (str, optional): Device ID. Defaults to None.
                fps (float, optional): Frames per second. Defaults to 30.0.
                resolution (List[float], optional): Resolution of the camera. Defaults to _480P.
                print_rgb_stereo_latency (bool, optional): Flag indicating whether to print RGB stereo latency. Defaults to False.
                show_disparity (bool, optional): Flag indicating whether to show disparity. Defaults to False.
                color_mode (str, optional): Color mode. Defaults to _BGR_MODE.
                auto_focus (bool, optional): Flag indicating whether to use auto focus. Defaults to True.
                get_depth (bool, optional): Flag indicating whether to get depth. Defaults to True.
                sync_depth (bool, optional): Flag indicating whether to synchronize depth. Defaults to True.
            Returns:
                None
        """

        print('Building RGBd Camera...')
        self.on = False
        self.cam_auto_mode = True
        self.device_id = device_id
        print(f'device_id: {self.device_id}')
        self.replay = replay
        self.fps = fps
        self.resolution = resolution
        self.print_rgb_stereo_latency=print_rgb_stereo_latency
        self.show_disparity=show_disparity
        if color_mode not in [self._RGB_MODE, self._BGR_MODE]:
            raise ValueError(f'color_mode must be one of {self._RGB_MODE} or {self._BGR_MODE}')
        self.color_mode = color_mode
        self.auto_focus = auto_focus
        self.get_depth = get_depth
        self.sync_depth = sync_depth
        
        self.rgb_fps_measured = 0
        self.depth_fps_measured = 0
        
        print(f'fps: {fps}, resolution: {resolution}')
        if self.replay :
            if replay_data is not None:
                self.load_replay(replay_data)
            self.next_frame = self.next_frame_video
            self.is_on = self.is_on_replay
            self.cam_data = cam_params
        else:
            self.cam_data = {}
            if self.get_depth:                
                self.mono_fps = 60 # maximum fps for mono cameras, to ensure that the stereo depth node can output with minimum latency compared to the RGB camera
                if not self.sync_depth:
                    self.next_frame = self.next_frame_depth_livestream
                else:
                    self.max_rgb_depth_latency = 20 # maximum latency between RGB and depth frames
                    self.next_frame = self.next_frame_depth_synced_livestream
            else:
                self.next_frame = self.next_frame_livestream
            self.build_device()
        self.rgb_frame = None
        self.depth_map = None
        self.new_frame = False
        self.rgb_timestamp = 0
        self.depth_timestamp = 0
        
        self.rgb_frames_thread = threading.Thread(target=self.collect_rgb_frames)
        self.rgb_frames_thread.start()
        if self.get_depth:
            self.depth_frames_thread = threading.Thread(target=self.collect_depth_frames)
            self.depth_frames_thread.start()    

        print(f'RGBd Camera built: replay={replay}')
    

    def build_device(self):
        """
        Builds the device and initializes camera settings.
        Args:
            None
        Returns:
            None
        Raises:
            None
        """
        
        print("Building device...")
        print(f'device_id: {self.device_id}')
        if self.device_id is None:
            self.device = dai.Device()
        else:
            self.device = dai.Device(dai.DeviceInfo(self.device_id), maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS)
            
        self.cam_data['resolution'] = (int(self.resolution[0]), int(self.resolution[1]))
        calibData = self.device.readCalibration()
        self.cam_data['matrix'] = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB, self.cam_data['resolution'][0], self.cam_data['resolution'][1]))
        self.cam_data['hfov'] = calibData.getFov(dai.CameraBoardSocket.RGB)
        
        if self.get_depth:
            if not self.sync_depth:
                self.create_rgb_depth_pipeline()
            else:
                self.create_rgb_depth_synced_pipeline(self.max_rgb_depth_latency)
        else:
            self.create_rgb_only_pipeline()

        print("Device built.")
    
    
    def create_RGB_pipeline(self):
        """
        Creates a pipeline for the RGB camera.
        Returns:
            dai.Pipeline: The created pipeline object.
        """
        
        self.rgb_res = dai.ColorCameraProperties.SensorResolution.THE_1080_P
        # print(f'rgb_res: {self.rgb_res[0]}x{self.rgb_res[1]}')
        print(f'resolution: {self.resolution}')
        
        print("Creating pipeline...")
        # Start defining a pipeline
        self.pipeline = dai.Pipeline()
        self.pipeline.setXLinkChunkSize(0) # decrease latency
        self.queue_size = 2
        self.blocking_queue = True
        
        # ColorCamera
        print("Creating Color Camera...")
        self.camRgb = self.pipeline.createColorCamera()
        self.camRgb.setResolution(self.rgb_res)
        
        # Set the properties of the RGB camera
        self.camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        self.camRgb.setCamera("color")
        
        if self.color_mode == self._RGB_MODE:
            self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
            print("Setting RGB color mode")
        else:
            self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            print("Setting BGR color mode")
        
        self.camRgb.setInterleaved(True)
        
        if self.resolution == self._720P:
            self.camRgb.setIspScale(2, 3)
        elif self.resolution == self._480P:
            self.camRgb.setIspScale(1, 3)
            
        
        if self.cam_auto_mode:
            print("Setting auto exposure and white balance")
            self.camRgb.initialControl.setAutoExposureEnable()
            self.camRgb.initialControl.setAutoExposureLimit(1000)
        else:
            self.expTime = 1000 # exposure time
            self.sensIso = 1200 # ISO sensitivity
            self.wbManual = 12000 # manual white balance
            self.camRgb.initialControl.setManualWhiteBalance(self.wbManual)
            self.camRgb.initialControl.setManualExposure(self.expTime, self.sensIso)
            print("Setting manual white balance: ", self.wbManual)
            print("Setting manual exposure time: ", self.expTime, "iso: ", self.sensIso)
            
        if not self.auto_focus:
            self.lensPos = 150 # lens position
            self.camRgb.initialControl.setManualFocus(self.lensPos)
        else:
            self.camRgb.initialControl.setAutoFocusMode(dai.RawCameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
        
        self.camRgb.setFps(self.fps)
        # camRgb.setPreviewSize(self.cam_data['resolution'][0], self.cam_data['resolution'][1])
        # camRgb.setVideoSize(self.cam_data['resolution'][0], self.cam_data['resolution'][1])
        
    def create_stereo_pipeline(self):
        """
        Creates a pipeline for the stereo cameras.
        Returns:
            dai.Pipeline: The created pipeline object.
        """
        
        self.mono_res = dai.MonoCameraProperties.SensorResolution.THE_400_P
        
        print("Creating Mono Cameras...")
        # Create left and right mono cameras
        camLeft = self.pipeline.create(dai.node.MonoCamera)
        camRight = self.pipeline.create(dai.node.MonoCamera)
        camLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        camRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)       
        
        # Set resolution and fps for mono camera
        for monoCam in (camLeft, camRight):
            monoCam.setResolution(self.mono_res)
            monoCam.setFps(self.mono_fps)
                

        # Create StereoDepth node that will produce the depth map
        self.stereo = self.pipeline.create(dai.node.StereoDepth)
        
        camLeft.out.link(self.stereo.left)
        camRight.out.link(self.stereo.right)

        self.stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
        self.stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        self.stereo.setLeftRightCheck(True)
        self.stereo.setExtendedDisparity(False)
        
        self.stereo.left.setBlocking(self.blocking_queue)
        self.stereo.left.setQueueSize(self.queue_size)
        self.stereo.right.setBlocking(self.blocking_queue)
        self.stereo.right.setQueueSize(self.queue_size)
        
    
    def create_rgb_only_pipeline(self):
        """
        Creates a pipeline for RGB-D cameras, but only RGB frames are captured.
        Returns:
            dai.Pipeline: The created pipeline object.
        """
        print("Creating pipeline...")
        # Start defining a pipeline
        self.create_RGB_pipeline()
        
        # Create XLinkOut node for RGB camera
        self.cam_out = self.pipeline.createXLinkOut()
        self.cam_out.setStreamName("rgb")
        self.cam_out.input.setQueueSize(self.queue_size)
        self.cam_out.input.setBlocking(self.blocking_queue)
        
        self.camRgb.isp.link(self.cam_out.input)
        # camRgb.video.link(self.cam_out.input)
        # camRgb.preview.link(self.cam_out.input)

        print("RGB pipeline creeeeeeeeeeeeeated.")
        self.device.startPipeline(self.pipeline)
        self.rgbQ = self.device.getOutputQueue(name="rgb", maxSize=self.queue_size, blocking=self.blocking_queue)
        
    def create_rgb_depth_pipeline(self):
        """
        Creates a pipeline for RGB-D cameras, captures RGB and depth frames asynchronously.
        Returns:
            dai.Pipeline: The created pipeline object.
        """
        self.create_RGB_pipeline()
        
        # Create XLinkOut node for RGB camera
        self.cam_out = self.pipeline.createXLinkOut()
        self.cam_out.setStreamName("rgb")
        self.cam_out.input.setQueueSize(self.queue_size)
        self.cam_out.input.setBlocking(self.blocking_queue)
        self.camRgb.isp.link(self.cam_out.input)
        
        self.create_stereo_pipeline()

        self.depth_out = self.pipeline.create(dai.node.XLinkOut)
        self.depth_out.setStreamName("depth")
        self.stereo.depth.link(self.depth_out.input)
        # decreases latency
        self.depth_out.input.setQueueSize(self.queue_size)
        self.depth_out.input.setBlocking(self.blocking_queue)
        
        print("RGB-depth (not synchronized) pipeline creeeeeeeeeeeeeated.")
        
        self.device.startPipeline(self.pipeline)
        self.rgbQ = self.device.getOutputQueue(name="rgb", maxSize=self.queue_size, blocking=self.blocking_queue)
        self.depthQ = self.device.getOutputQueue(name="depth", maxSize=self.queue_size, blocking=self.blocking_queue)

    def create_rgb_depth_synced_pipeline(self, max_rgb_depth_latency):
        """
        Creates a pipeline for RGB-D cameras, captures RGB and depth frames synchronously, within a given latency tolerance.
        Returns:
            dai.Pipeline: The created pipeline object.
        """
        print("Creating pipeline...")
        # Start defining a pipeline
        self.create_RGB_pipeline()
        
        self.create_stereo_pipeline()
        

        sync = self.pipeline.create(dai.node.Sync)
        sync.setSyncThreshold(timedelta(milliseconds=max_rgb_depth_latency))
        xoutGrp = self.pipeline.create(dai.node.XLinkOut)

        xoutGrp.setStreamName("xout")
        xoutGrp.input.setBlocking(self.blocking_queue)
        xoutGrp.input.setQueueSize(self.queue_size)
        
        self.stereo.depth.link(sync.inputs["depth"])      
        self.camRgb.isp.link(sync.inputs["rgb"])
        
        sync.out.link(xoutGrp.input)
        
        print("RGB-depth (synchronized) pipeline creeeeeeeeeeeeeated.")
        self.device.startPipeline(self.pipeline)
        self.synced_queue = self.device.getOutputQueue(name="xout", maxSize=self.queue_size, blocking=self.blocking_queue)

    def load_replay(self, replay):
        """
        Load a replay file.

        Parameters:
        - replay (dict): A dictionary containing the replay information.

        Raises:
        - ValueError: If there is an error reading the video.

        """
        self.video = cv2.VideoCapture(replay['Video'])
        self.nb_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_index = 0
        if not self.video.isOpened():
            raise ValueError("Error reading video")
        self.timestamps = replay['Timestamps']
        self.depth_maps = replay['Depth_maps']
        
    def collect_rgb_frames(self):
        """
        Collects RGB frames from the camera.

        Returns:
            None
        """
        old_rgb_timestamp = 1
        
        while True:
            if self.on:
                try:
                    frame = self.rgbQ.get()
                    self.rgb_timestamp = frame.getTimestamp().total_seconds()
                    self.rgb_fps_measured = int(1.0 / (self.rgb_timestamp - old_rgb_timestamp))
                    self.rgb_frame = frame.getCvFrame()
                    self.new_frame = True
                    old_rgb_timestamp = self.rgb_timestamp  # update old timestamp for next frame
                except:
                    self.rgb_frame = None
            else:
                time.sleep(0.05)
    
    def collect_depth_frames(self):
        """
        Collects depth frames from the camera.

        Returns:
            None
        """
        old_depth_timestamp = 1
        while True:
            if self.on:
                try:
                    frame = self.depthQ.get()
                    self.depth_timestamp = frame.getTimestamp().total_seconds() 
                    self.depth_map = frame.getFrame()

                    self.depth_fps_measured = int(1.0 / (self.depth_timestamp - old_depth_timestamp))
                    old_depth_timestamp = self.depth_timestamp  # update old timestamp for next frame
                except:
                    self.depth_map = None
            else:
                time.sleep(0.05)
        
    def next_frame_livestream(self):
        """
        Retrieves the next frame from the livestream, without depth map.
        Returns:
            Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]: A tuple containing the success status, the RGB frame, and the depth map.
                - success (bool): True if both the RGB frame and the depth map are available, False otherwise.
                - frame (Optional[np.ndarray]): The RGB frame as a NumPy array, or None if not available.
        """
        try:
            r_frame = self.rgbQ.get()
            self.rgb_timestamp = r_frame.getTimestamp().total_seconds()
            
        except:
            return False, None, None, None
        
        if r_frame is not None:
            frame = r_frame.getCvFrame()
            frame = cv2.resize(frame, self.cam_data['resolution'])
            self.rgb_frame = frame
            self.new_frame = True
            success = True
        else:
            self.rgb_frame = None
            success = False
            
        
        if success and self.print_rgb_stereo_latency:
            now = dai.Clock.now()
            rgb_latency = (now - r_frame.getTimestamp()).total_seconds() * 1000
            print(f'rgb latency: {rgb_latency} ms')
        
        if success:
            return success, self.rgb_frame, None, self.rgb_timestamp
        else:
            print('unsuccessful')
            return False, None, None, None
        
    def next_frame_depth_livestream(self):
        """
        Retrieves the next frame from the livestream.
        Returns:
            Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]: A tuple containing the success status, the RGB frame, and the depth map.
                - success (bool): True if both the RGB frame and the depth map are available, False otherwise.
                - frame (Optional[np.ndarray]): The RGB frame as a NumPy array, or None if not available.
                - depth_map (Optional[np.ndarray]): The depth map as a NumPy array, or None if not available.
        """
        
        if self.rgb_frame is not None and self.depth_map is not None:
            success = True
        else:
            success = False
            
        if success and self.print_rgb_stereo_latency:
            now = dai.Clock.now().total_seconds()
            rgb_latency = (now - self.rgb_timestamp) * 1000
            depth_latency = (now - self.depth_timestamp) * 1000
            rgb_depth_delay = (self.depth_timestamp - self.rgb_timestamp) * 1000

            print(f'fps rgb: {self.rgb_fps_measured}')
            print(f'fps depth: {self.depth_fps_measured}')
            print(f'rgb latency: {rgb_latency} ms')
            print(f'depth latency: {depth_latency} ms')
            print(f'rgb-depth delay: {rgb_depth_delay} ms')
            
        if success and self.show_disparity:
            depthFrameColor = cv2.normalize(self.depth_map, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)
            cv2.imshow(f'depth {self.device_id}', depthFrameColor)
            cv2.waitKey(1)
        if success:
            return success, self.rgb_frame, self.depth_map, self.rgb_timestamp
        else:
            print('unsuccessful')
            return success, None, None, None
    
    def next_frame_depth_synced_livestream(self):
        """
        Retrieves the next frame from the livestream. The RGB and depth frames are synchronized. (at the cost of latency ~ 200ms)
        Returns:
            Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]: A tuple containing the success status, the RGB frame, and the depth map.
                - success (bool): True if both the RGB frame and the depth map are available, False otherwise.
                - frame (Optional[np.ndarray]): The RGB frame as a NumPy array, or None if not available.
                - depth_map (Optional[np.ndarray]): The depth map as a NumPy array, or None if not available.
        """
        try:
            frames = self.synced_queue.get()
            
            r_frame = frames['rgb']
            d_frame = frames['depth']
            
            self.rgb_timestamp = r_frame.getTimestamp()
            
        except:
            # raise 
            return False, None, None, None
        
        if r_frame is not None:
            frame = r_frame.getCvFrame()
            # frame = cv2.resize(frame, self.cam_data['resolution'])
            self.rgb_frame = frame
            self.new_frame = True
        else:
            self.rgb_frame = None
            
        if d_frame is not None:
            frame = d_frame.getFrame()
            frame = cv2.resize(frame, self.cam_data['resolution'])
            self.depth_map = frame
        else:
            self.depth_map = None


        if self.show_disparity:
            depthFrameColor = cv2.normalize(self.depth_map, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)
            cv2.imshow(f'depth {self.device_id}', depthFrameColor)
            cv2.waitKey(1)

        if self.print_rgb_stereo_latency:
            now = dai.Clock.now()
            rgb_latency = (now - r_frame.getTimestamp()).total_seconds() * 1000
            depth_latency = (now - d_frame.getTimestamp()).total_seconds() * 1000
            print(f'rgb latency: {rgb_latency} ms, depth latency: {depth_latency} ms')
            rgb_depth_delay = (d_frame.getTimestamp() - r_frame.getTimestamp()).total_seconds() * 1000
            print(f'rgb-depth delay: {rgb_depth_delay} ms')
        
        if self.rgb_frame is not None and self.depth_map is not None:
            success = True
        else:
            success = False
            
        if success:
            return success, self.rgb_frame, self.depth_map, self.rgb_timestamp
        else:
            return False, None, None, None
    
    
    def next_frame_video(self):
        """
        Reads the next frame from the video and updates the current frame, depth map, and timestamp.

        Returns:
            Tuple[bool, np.ndarray, np.ndarray]: A tuple containing the success status of reading the frame,
            the frame itself, and the corresponding depth map.
        """
        success, frame = self.video.read()
        self.rgb_frame = frame
        self.depth_map = self.depth_maps[self.current_frame_index]
        self.rgb_timestamp = self.timestamps[self.current_frame_index]
        self.current_frame_index += 1
        self.new_frame = True
        return success, self.rgb_frame, self.depth_map
    
    def get_depth_map(self):
        """
        Get the current depth map.

        Returns:
            np.ndarray: The depth map as a NumPy array.
        """
        return self.depth_map

    def start(self):
        """
        Start the RGB-D camera.

        Returns:
            None
        """
        self.on = True
    
    def stop(self):
        """
        Stop the RGB-D camera.

        Returns:
            None
        """
        self.on = False
        #wait for 50ms
        if not self.replay:
            time.sleep(0.05)
            self.device.close()
        self.rgb_frames_thread.join()
        if self.get_depth:
            self.depth_frames_thread.join()

    def get_res(self):
        """
        Get the resolution of the RGB-D camera.

        Returns:
            Tuple[int, int]: The resolution as a tuple of width and height.
        """
        return self.cam_data['resolution']
    
    def get_device_data(self):
        """
        Get the device data of the RGB-D camera.

        Returns:
            Dict: The device data as a dictionary.
        """
        return self.cam_data

    def get_num_frames(self):
        """
        Get the total number of frames in the video replay.

        Returns:
            int: The number of frames.
        """
        return self.nb_frames
    
    def get_timestamps(self):
        """
        Get the timestamps of the frames in the video replay.

        Returns:
            List[float]: The timestamps as a list of floats.
        """
        return self.timestamps
    
    
    def is_on(self):
        """
        Check if the RGB-D camera is currently on.

        Returns:
            bool: True if the camera is on, False otherwise.
        """
        return self.on
    
    def is_on_replay(self):
        """
        Check if the RGB-D camera is currently on replay mode.

        Returns:
            bool: True if the camera is on replay mode, False otherwise.
        """
        return self.current_frame_index < self.nb_frames