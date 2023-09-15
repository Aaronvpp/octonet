import pyrealsense2 as rs
import numpy as np
import cv2
import os
import pickle

# Create output directories
output_directory = os.path.dirname(os.path.abspath(__file__))
index = 0
while True:
    main_output_folder = f'output_{index}'
    main_output_path = os.path.join(output_directory, main_output_folder)
    depth_output_folder = f'depth_image_output_{index}'
    depth_output_path = os.path.join(main_output_path, depth_output_folder)
    rgb_output_folder = f'rgb_video_output_{index}'
    rgb_output_path = os.path.join(main_output_path, rgb_output_folder)
    if not os.path.exists(main_output_path):
        os.makedirs(main_output_path)
        os.makedirs(depth_output_path)
        os.makedirs(rgb_output_path)
        break
    index += 1

# Define the codec and create VideoWriter objects
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
rgb_video_filename = os.path.join(rgb_output_path, 'rgb_video.mp4')
rgb_video_writer = cv2.VideoWriter(rgb_video_filename, fourcc, 30, (640, 480))

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

frame_number = 0

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Write the RGB frame to the MP4 video file
        rgb_video_writer.write(color_image)

        # Convert the depth_image to grayscale
        depth_image_gray = cv2.convertScaleAbs(depth_image, alpha=0.03)

        # Save the grayscale depth_image as a pickle file
        pickle_filename = f'depth_image_{frame_number:04d}.pkl'
        pickle_filepath = os.path.join(depth_output_path, pickle_filename)
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(depth_image_gray, f)

        # Increment the frame number
        frame_number += 1

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', np.hstack((color_image, depth_colormap)))
        cv2.waitKey(1)

    # Release the video writer
    rgb_video_writer.release()

finally:

    # Stop streaming
    pipeline.stop()