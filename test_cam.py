
from RgbdCameras2 import SimpleRgbdCam as RgbdCamera
import cv2

if __name__ == "__main__":
    # rgbd_camera = RgbdCamera(resolution=(1280,720), fps=60.0, auto_focus=True, get_depth=True, sync_depth=False, print_rgb_stereo_latency=True, show_disparity=False)
    rgbd_camera = RgbdCamera(resolution=(1280,720), fps=60.0, auto_focus=True, get_depth=True, sync_depth=False, print_rgb_stereo_latency=True, show_disparity=False)
    rgbd_camera.start()
    while rgbd_camera.is_on():
        success, img, map, rgb_timestamp, depth_timestamp = rgbd_camera.get_frame()
        if not success:
            print("Failed to get frame.")
            continue
        cv2.imshow('img', img)
        # cv2.imshow('map', map)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    rgbd_camera.stop()
    cv2.destroyAllWindows()
    print("End of test.")