from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera

# Create the camera configuration
camera_config = OpenCVCameraConfig(camera_index=0, fps=30, width=640, height=480)

# Instantiate the camera
camera = OpenCVCamera(camera_config)

# Connect the camera (required before using it)
camera.connect()

# Now you can read frames from the camera
frame = camera.read()