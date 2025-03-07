import depthai as dai
import cv2
import numpy as np
import time
from typing import Optional, Any, Dict, List
from datetime import timedelta

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
                self.mono_fps = 40 # maximum fps for mono cameras, to ensure that the stereo depth node can output with minimum latency compared to the RGB camera
                if not self.sync_depth:
                    self.next_frame = self.next_frame_depth_livestream
                else:
                    self.max_rgb_depth_latency = 20 # maximum latency between RGB and depth frames
                    self.next_frame = self.next_frame_depth_synced_livestream
            else:
                self.next_frame = self.next_frame_livestream
            self.build_device()
        self.frame = None
        self.new_frame = False
        self.rgb_timestamp = 0
        self.depth_timestamp = 0

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
        
        # ColorCamera
        print("Creating Color Camera...")
        self.camRgb = self.pipeline.createColorCamera()
        self.camRgb.setResolution(self.rgb_res)
        
        # Set the properties of the RGB camera
        self.camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
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
        # cam.setIspScale(self.scale_nd[0], self.scale_nd[1])
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
        self.stereo.initialConfig.setConfidenceThreshold(245)
        self.stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        
        camLeft.out.link(self.stereo.left)
        camRight.out.link(self.stereo.right)

        # Closer-in minimum depth, disparity range is doubled (from 95 to 190):
        extended_disparity = False
        # Better accuracy for longer distance, fractional disparity 32-levels:
        subpixel = False
        # Better handling for occlusions:
        lr_check = True

        self.stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        
        self.stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7) # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
        self.stereo.setLeftRightCheck(lr_check)
        self.stereo.setExtendedDisparity(extended_disparity)
        self.stereo.setSubpixel(subpixel)
    
    
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
        self.cam_out.input.setQueueSize(1)
        self.cam_out.input.setBlocking(False)
        self.camRgb.isp.link(self.cam_out.input)
        # camRgb.video.link(self.cam_out.input)
        # camRgb.preview.link(self.cam_out.input)

        print("RGB pipeline creeeeeeeeeeeeeated.")
        self.device.startPipeline(self.pipeline)
        self.rgbQ = self.device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        
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
        self.cam_out.input.setQueueSize(1)
        self.cam_out.input.setBlocking(False)
        self.camRgb.isp.link(self.cam_out.input)
        
        self.create_stereo_pipeline()

        self.depth_out = self.pipeline.create(dai.node.XLinkOut)
        self.depth_out.setStreamName("depth")
        self.stereo.depth.link(self.depth_out.input)
        # decreases latency
        self.depth_out.input.setQueueSize(1)
        self.depth_out.input.setBlocking(False)
        
        print("RGB-depth (not synchronized) pipeline creeeeeeeeeeeeeated.")
        
        self.device.startPipeline(self.pipeline)
        self.rgbQ = self.device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        self.depthQ = self.device.getOutputQueue(name="depth", maxSize=1, blocking=False)

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
        
        self.stereo.depth.link(sync.inputs["depth"])      
        self.camRgb.isp.link(sync.inputs["rgb"])
        
        sync.out.link(xoutGrp.input)
        
        print("RGB-depth (synchronized) pipeline creeeeeeeeeeeeeated.")
        self.device.startPipeline(self.pipeline)
        self.synced_queue = self.device.getOutputQueue(name="xout", maxSize=1, blocking=False)

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
            self.frame = frame
            self.new_frame = True
            success = True
        else:
            self.frame = None
            success = False
            
        
        if self.print_rgb_stereo_latency:
            now = dai.Clock.now()
            rgb_latency = (now - r_frame.getTimestamp()).total_seconds() * 1000
            print(f'rgb latency: {rgb_latency} ms')
        
        if success:
            return success, self.frame, None, self.rgb_timestamp
        else:
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
        try:
            t = time.time()
            r_frame = self.rgbQ.get()
            print(f'rgb elapsed: {(time.time()-t)*1000} ms')
            t = time.time()
            d_frame = self.depthQ.get()
            print(f'depth elapsed: {(time.time()-t)*1000} ms')
            
        except:
            return False, None, None, None
        
        if r_frame is not None:
            frame = r_frame.getCvFrame()
            new_rgb_timestamp = r_frame.getTimestamp().total_seconds()
            fps_rgb = 1/(new_rgb_timestamp - self.rgb_timestamp)
            self.rgb_timestamp = new_rgb_timestamp
            # frame = cv2.resize(frame, self.cam_data['resolution'])
            self.frame = frame
            self.new_frame = True
        else:
            self.frame = None
            
        if d_frame is not None:
            frame = d_frame.getFrame()
            new_depth_timestamp = d_frame.getTimestamp().total_seconds()
            fps_depth = 1/(new_depth_timestamp - self.depth_timestamp)
            self.depth_timestamp = new_depth_timestamp
            frame = cv2.resize(frame, self.cam_data['resolution'])
            self.depth_map = frame
        else:
            self.depth_map = None


        if self.print_rgb_stereo_latency:
            now = dai.Clock.now().total_seconds()
            rgb_latency = (now - self.rgb_timestamp) * 1000
            depth_latency = (now - self.depth_timestamp) * 1000
            rgb_depth_delay = (self.depth_timestamp - self.rgb_timestamp) * 1000

            print(f'fps rgb: {fps_rgb}')
            print(f'fps depth: {fps_depth}')
            print(f'rgb latency: {rgb_latency} ms')
            print(f'depth latency: {depth_latency} ms')
            print(f'rgb-depth delay: {rgb_depth_delay} ms')
            
        if self.show_disparity:
            depthFrameColor = cv2.normalize(self.depth_map, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)
            cv2.imshow(f'depth {self.device_id}', depthFrameColor)
            cv2.waitKey(1)

        
        if self.frame is not None and self.depth_map is not None:
            success = True
        else:
            success = False
        if success:
            return success, self.frame, self.depth_map, self.rgb_timestamp
        else:
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
            self.frame = frame
            self.new_frame = True
        else:
            self.frame = None
            
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
        
        if self.frame is not None and self.depth_map is not None:
            success = True
        else:
            success = False
            
        if success:
            return success, self.frame, self.depth_map, self.rgb_timestamp
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
        self.frame = frame
        self.depth_map = self.depth_maps[self.current_frame_index]
        self.rgb_timestamp = self.timestamps[self.current_frame_index]
        self.current_frame_index += 1
        self.new_frame = True
        return success, self.frame, self.depth_map
    
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