import depthai as dai
import numpy as np
import cv2
import time
from datetime import timedelta

fps = 40

queue_size = 3
blocking_queue = True

pipeline = dai.Pipeline()
pipeline.setXLinkChunkSize(0)

monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")

monoLeft.setFps(fps)
monoRight.setFps(fps)

color = pipeline.create(dai.node.ColorCamera)
color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
color.setCamera("color")
color.setFps(fps)

stereo = pipeline.create(dai.node.StereoDepth)
stereo.setLeftRightCheck(True)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)

stereo.left.setBlocking(blocking_queue)
stereo.left.setQueueSize(queue_size)
stereo.right.setBlocking(blocking_queue)
stereo.right.setQueueSize(queue_size)

xoutGrp = pipeline.create(dai.node.XLinkOut)
xoutGrp.setStreamName("xout")
xoutGrp.input.setBlocking(blocking_queue)
xoutGrp.input.setQueueSize(queue_size)



sync = pipeline.create(dai.node.Sync)
sync.setSyncThreshold(timedelta(milliseconds=20))

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

stereo.disparity.link(sync.inputs["disparity"])
color.video.link(sync.inputs["video"])

sync.out.link(xoutGrp.input)

disparityMultiplier = 255.0 / stereo.initialConfig.getMaxDisparity()

with dai.Device(pipeline) as device:
    queue = device.getOutputQueue("xout", queue_size, blocking_queue)
    fps = {}
    timestamps = {}
    timestamps["video"] = 0
    timestamps["disparity"] = 0
    t_loop = time.time()
    while True:
        t_new = time.time()
        if t_new - t_loop > 0:
            print(f"FPS: {int(1/(t_new-t_loop))}")
            t_loop = t_new
        msgGrp = queue.get()
        for name, msg in msgGrp:
            frame = msg.getCvFrame()
            new_timestamp = msg.getTimestamp().total_seconds()
            fps[name] = int(1.0 / (new_timestamp - timestamps[name]))
            timestamps[name] = new_timestamp
            print(f"{name} FPS: {fps[name]}")
            if name == "disparity":
                frame = (frame * disparityMultiplier).astype(np.uint8)
                frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
                
            cv2.imshow(name, frame)
        if cv2.waitKey(1) == ord("q"):
            break
