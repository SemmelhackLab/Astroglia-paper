"""
Video Clipper and Annotation Tools

This module provides utility functions for processing video files, including:
1. Creating videos from image sequences.
2. Adding text annotations to video frames.
3. Extracting frames from videos.
4. Cutting video clips based on behavioral bouts.
5. Superimposing tracking annotations (head, eyes, jaw) onto videos.

Dependencies:
- cv2 (OpenCV)
- numpy
- h5py
- os

Author: [Your Name/Organization]
Date: [Current Date]
"""

import os
import cv2
import numpy as np
import h5py

# Constants for file paths
VISUAL_FIST_SIFT_DIR = "D:/visual_fist_sift"
EXTRACTED_FRAMES_DIR = os.path.join(VISUAL_FIST_SIFT_DIR, "extracted_frames")
ANNOTATED_VIDEO_DIR = os.path.join(VISUAL_FIST_SIFT_DIR, "annotated_visual_videos")
SLEAP_TRACKED_DIR = os.path.join(VISUAL_FIST_SIFT_DIR, "sleap_tracked_results_h5")


def create_video_from_images(image_folder, output_video_path, bout, fps=200, coord_path=None, center=None,
                             report=False, expand=False, cropping=False, crop_size=200, video_name=None, text=False):
    """
    Creates a video file from a sequence of images.

    Args:
        image_folder (str): Path to the folder containing images.
        output_video_path (str): Path where the output video will be saved.
        bout (list or tuple): [start_frame, end_frame] range of images to include.
        fps (int): Frames per second for the output video. Default is 200.
        coord_path (str, optional): Path to coordinate files for cropping.
        center (int, optional): Center frame index (used for naming if provided).
        report (bool): If True, prints status messages.
        expand (bool): If True, expands the range by 1 second (fps) on both sides.
        cropping (bool): If True, crops the video around the head position.
        crop_size (int): Size of the crop window (radius). Default is 200.
        video_name (str, optional): Name of the video (used for loading coordinates).
        text (bool): If True, adds 'Bout' text annotation to frames within the bout range.
    """
    # Get a list of image file names in the specified folder
    if not os.path.exists(image_folder):
        print(f"Error: Image folder not found: {image_folder}")
        return

    total_frames = int(len(os.listdir(image_folder)))

    if expand:
        start_frame = int(max(0, bout[0] - (1 * fps)))
        end_frame = int(min(bout[1] + (1 * fps), total_frames - 1))
    else:
        start_frame = bout[0]
        end_frame = bout[1]

    images = [str(i) + '.jpg' for i in range(start_frame, end_frame)]

    # Check if there are images to process
    if not images:
        print("No images found in the specified folder.")
        return

    if not os.path.exists(output_video_path):
        os.makedirs(output_video_path)

    if center is not None:
        saved_file_name = os.path.join(output_video_path, f'center_{center}.mp4')
    else:
        saved_file_name = os.path.join(output_video_path, f'{bout[0]}_{bout[1]}.mp4')

    # Read the first image to get the width and height
    first_image_path = os.path.join(image_folder, images[0])
    if not os.path.exists(first_image_path):
        print(f"Error: First image not found: {first_image_path}")
        return

    first_image = cv2.imread(first_image_path)
    if first_image is None:
        print(f"Error loading first image: {first_image_path}")
        return

    height, width, layers = first_image.shape
    
    # Initialize crop parameters
    y_start, y_end, x_start, x_end = 0, height, 0, width

    if cropping:
        if coord_path is None or video_name is None:
            print("Error: coord_path and video_name are required for cropping.")
            return

        try:
            head_index_x = np.load(os.path.join(coord_path, f'{video_name}_full_head_x.npy'))
            head_index_y = np.load(os.path.join(coord_path, f'{video_name}_full_head_y.npy'))
            crop_center = [int(head_index_x[start_frame]), int(head_index_y[start_frame])]

            y_start = max(0, crop_center[1] - crop_size)
            y_end = min(height, crop_center[1] + crop_size)
            x_start = max(0, crop_center[0] - crop_size)
            x_end = min(width, crop_center[0] + crop_size)

            # Adjust crop boundaries if they hit the edges
            if y_start == 0:
                y_end += abs(crop_center[1] - crop_size)
            if x_start == 0:
                x_end += abs(crop_center[0] - crop_size)
            if y_end == height:
                y_start -= abs(crop_center[1] + crop_size - height)
            if x_end == width:
                x_start -= abs(crop_center[0] + crop_size - width)
                
            frame_size = (crop_size * 2, crop_size * 2)
        except Exception as e:
            print(f"Error loading coordinates for cropping: {e}")
            return
    else:
        frame_size = (width, height)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    out = cv2.VideoWriter(saved_file_name, fourcc, fps, frame_size)

    # Loop through the images and write them to the video
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)

        # Check if the frame was loaded successfully
        if frame is None:
            print(f"Error loading image {image_path}")
            continue

        if cropping:
            frame = frame[y_start:y_end, x_start:x_end]

        if text:
            # Check if frame index is within the bout range
            frame_idx = int(image_name.split('.')[0])
            if bout[0] <= frame_idx <= bout[1]:
                cv2.putText(frame, 'Bout', (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
        
        out.write(frame)  # Write the frame to the video

    # Release the VideoWriter object
    out.release()
    if report:
        print(f"Video saved as {saved_file_name}")


def add_text_to_frames(input_video_path, output_video_path, frame_indices, text, report=False):
    """
    Adds text annotation to specific frames in a video.

    Args:
        input_video_path (str): Path to the input video.
        output_video_path (str): Path where the annotated video will be saved.
        frame_indices (list): List of frame indices to add text to.
        text (str): Text to add to the frames.
        report (bool): If True, prints status messages.
    """
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    current_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if there are no frames to read

        # If the current frame is in the specified indices, add text
        if current_frame in frame_indices:
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 4, cv2.LINE_AA)

        # Write the (possibly modified) frame to the new video
        out.write(frame)
        current_frame += 1

    # Release everything
    cap.release()
    out.release()
    if report:
        print(f"Processed video saved as {output_video_path}")


def extract_frames(input_video_path, output_folder):
    """
    Extracts all frames from a video and saves them as images.

    Args:
        input_video_path (str): Path to the input video.
        output_folder (str): Directory to save extracted images.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the input video
    cap = cv2.VideoCapture(input_video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if there are no frames to read

        # Save the frame as an image
        frame_filename = os.path.join(output_folder, str(frame_count) + ".jpg")  # Save as JPEG
        cv2.imwrite(frame_filename, frame)
        if frame_count % 100 == 0:
            print(f"Saved frame {frame_count}")

        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Extracted {frame_count} frames and saved to {output_folder}")


def create_video_with_text(image_folder, output_video_path, start, end, text, bouts, fps=100, center=None):
    """
    Creates a video from images with text annotations based on bout intervals.

    Args:
        image_folder (str): Path to folder containing images.
        output_video_path (str): Directory to save the output video.
        start (int): Start frame index.
        end (int): End frame index.
        text (str): Base text to annotate.
        bouts (list of lists): List of [start, end] bout intervals.
        fps (int): Frames per second.
        center (int, optional): Center frame index (for naming).
    """
    if not os.path.exists(output_video_path):
        os.makedirs(output_video_path)

    # Pre-calculate which frames need text and what text suffix
    text_frame_indices = []
    
    for i in range(len(bouts)):
        bout_start, bout_end = bouts[i]
        
        # Check intersection between bout and requested range [start, end]
        intersect_start = max(start, bout_start)
        intersect_end = min(end, bout_end)
        
        if intersect_start <= intersect_end:
            # Create list of frames for this bout within the range
            frames_in_bout = list(range(intersect_start, intersect_end + 1 if intersect_end < end else end))
            text_frame_indices.append(frames_in_bout)

    # Get a list of image file names in the specified folder
    images = [str(i) + '.jpg' for i in range(start, end)]

    # Check if there are images to process
    if not images:
        print("No images found in the specified folder.")
        return

    # Read the first image to get the width and height
    first_image_path = os.path.join(image_folder, images[0])
    if not os.path.exists(first_image_path):
        print(f"Error: First image not found: {first_image_path}")
        return
        
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    
    if center is not None:
        final_path = os.path.join(output_video_path, f'center_{center}.mp4')
    else:
        final_path = os.path.join(output_video_path, f'{start}_{end}.mp4')
        
    out = cv2.VideoWriter(final_path, fourcc, fps, (width, height))

    # Loop through the images and write them to the video
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)

        # Check if the frame was loaded successfully
        if frame is None:
            print(f"Error loading image {image_path}")
            continue

        frame_idx = int(image_name.split('.')[0])
        
        for j in range(len(text_frame_indices)):
            if frame_idx in text_frame_indices[j]:
                text_to_put = f"{text}_{j + 1}"
                cv2.putText(frame, text_to_put, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        out.write(frame)  # Write the frame to the video

    # Release the VideoWriter object
    out.release()
    print(f"Video saved as {final_path}")


def cut_video_clips(video_path, top_output_path, bouts, report=False):
    """
    Cuts clips from a video based on bout intervals.

    Args:
        video_path (str): Path to the source video.
        top_output_path (str): Directory to save extracted clips.
        bouts (list of lists): List of [start_frame, end_frame] intervals.
        report (bool): If True, prints status messages.
    """
    if not os.path.exists(top_output_path):
        os.makedirs(top_output_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
        
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for i, (start, end) in enumerate(bouts):
        # Set the starting frame (with 1 second buffer)
        start_frame = int(max(0, start - (1 * fps)))
        end_frame = int(min(end + (1 * fps), total_frames - 1))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
        output_file = os.path.join(top_output_path, f'{start}_{end}.mp4')
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        # Read and write frames from start to end
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break  # Exit if no frames are returned

            if start <= frame_idx <= end:
                cv2.putText(frame, 'bout', (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
            out.write(frame)

        # Release the VideoWriter object
        out.release()
        if report:
            print(f'Saved clip {i + 1}: {output_file}')

    # Release the video capture object
    cap.release()


def superimpose_annotation(video_name, fps=200, ranges=None, output_path=None):
    """
    Superimposes tracking annotations (head, eyes, jaw) onto video frames.

    Args:
        video_name (str): Name of the video dataset.
        fps (int): Frames per second for the output video.
        ranges (list, optional): [start_frame, end_frame] range to process.
        output_path (str, optional): Directory to save the output video.
    """
    print(f'Started superimposing: {video_name}')
    
    # Define paths
    image_folder = os.path.join(EXTRACTED_FRAMES_DIR, video_name)
    
    if ranges is None:
        name = video_name
    else:
        name = f'{video_name}_{ranges[0]}_{ranges[1]}.mp4'
        
    if output_path is None:
        output_video_path = os.path.join(ANNOTATED_VIDEO_DIR, name)
    else:
        output_video_path = os.path.join(output_path, name)

    if not os.path.exists(image_folder):
        print(f"Error: Image folder not found: {image_folder}")
        return

    total_frames = int(len(os.listdir(image_folder)))

    if ranges:
        image_files = [f'{i}.jpg' for i in range(ranges[0], min(ranges[1] + 1, total_frames))]
    else:
        image_files = [f'{i}.jpg' for i in range(total_frames)]

    h5_file_path = os.path.join(SLEAP_TRACKED_DIR, video_name + '.h5')
    if not os.path.exists(h5_file_path):
         print(f"Error: H5 file not found: {h5_file_path}")
         return

    with h5py.File(h5_file_path, 'r') as f:
        tracks_matrix = f['tracks'][:]

    # Extract coordinates
    # Shape of tracks_matrix is typically (frames, nodes, 2, instance) or similar
    # Adjusting based on original code usage: tracks_matrix[:,0,0,:].T.flatten()
    head_up_x = tracks_matrix[:, 0, 0, :].T.flatten()
    head_up_y = tracks_matrix[:, 1, 0, :].T.flatten()

    eye_mid_x = tracks_matrix[:, 0, 1, :].T.flatten()
    eye_mid_y = tracks_matrix[:, 1, 1, :].T.flatten()

    jaw_front_x = tracks_matrix[:, 0, 2, :].T.flatten()
    jaw_front_y = tracks_matrix[:, 1, 2, :].T.flatten()

    jaw_btm_x = tracks_matrix[:, 0, 3, :].T.flatten()
    jaw_btm_y = tracks_matrix[:, 1, 3, :].T.flatten()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For .mp4
    video_writer = None
    
    # Process each image
    count = ranges[0] if ranges else 0
    if ranges:
        print(f"Processing range starting at {count}")

    for img_file in image_files:
        # Load the image
        temp_image_path = os.path.join(image_folder, img_file)
        image = cv2.imread(temp_image_path)

        # Check if image is loaded successfully
        if image is None:
            print(f"Error loading image: {img_file}")
            count += 1
            continue

        # Initialize video writer with the first image dimensions
        if video_writer is None:
            frame_height, frame_width, _ = image.shape
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        try:
            coord_head_up = (int(head_up_x[count]), int(head_up_y[count]))
            coord_eye_mid = (int(eye_mid_x[count]), int(eye_mid_y[count]))
            coord_jaw_front = (int(jaw_front_x[count]), int(jaw_front_y[count]))
            coord_jaw_btm = (int(jaw_btm_x[count]), int(jaw_btm_y[count]))

            cv2.circle(image, coord_head_up, radius=5, color=(0, 0, 255), thickness=1)
            cv2.circle(image, coord_eye_mid, radius=5, color=(0, 0, 255), thickness=1)
            cv2.circle(image, coord_jaw_front, radius=5, color=(0, 0, 255), thickness=1)
            cv2.circle(image, coord_jaw_btm, radius=5, color=(0, 0, 255), thickness=1)
            
            video_writer.write(image)
        except Exception as e:
            print(f'Error processing frame {count}: {e}')
            video_writer.write(image)

        count += 1
        if count % 100 == 0:
            print(f"Processed {count} frames")

    # Release the video writer
    if video_writer:
        video_writer.release()
        print(f"Video saved as {output_video_path}")
    else:
        print("No video created (no frames processed).")


if __name__ == "__main__":
    print("This module provides video processing utilities.")
    print("Import this module to use the functions.")
