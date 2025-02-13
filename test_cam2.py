
from RgbdCameras2 import SimpleRgbdCam as RgbdCamera
import time
import numpy as np  

if __name__ == "__main__":
    # rgbd_camera = RgbdCamera(resolution=(1280,720), fps=60.0, auto_focus=True, get_depth=True, sync_depth=False, print_rgb_stereo_latency=True, show_disparity=False)
    computer_fps_list = []
    rgbd_camera = RgbdCamera( fps_rgb=60.0, show_fps=False, show_stats=True)
    rgbd_camera.start()
    while rgbd_camera.is_on():
        t = time.time()
        success, img, map, rgb_timestamp, depth_timestamp = rgbd_camera.next_frame()
        if not success:
            # print("Failed to get frame.")
            continue
        # cv2.imshow('img', img)
        # cv2.imshow('map', map)  # Uncomment this line to show the map if needed
        # print(f"RGB timestamp: {rgb_timestamp}, Depth timestamp: {depth_timestamp}")  # Print timestamps for debugging  
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
    print("End of test.")