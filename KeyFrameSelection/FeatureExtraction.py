import os
import csv
import pandas as pd
import av
import cv2

def _get_timestamp(frame_idx, fps):
    """
    Converts a frame index to a formatted timestamp string (HH:MM:SS.mmm).

    Args:
        frame_idx (int): Index of the frame in the video.
        fps (float): Frames per second of the video.

    Returns:
        str: Timestamp in the format 'HH:MM:SS.mmm' representing the frame time.
    """
    seconds = frame_idx / fps
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def process_video(video_path, interval_sec=3):
    """
    Samples frames from a video at fixed time intervals.

    Args:
        video_path (str): Path to the input video file.
        interval_sec (int, optional): Time interval in seconds between sampled frames. Defaults to 3.

    Returns:
        tuple:
            - records (list): List of tuples (frame, frame_idx) for each sampled frame.
            - fps (float): Frames per second of the input video.
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    interval = int(fps * interval_sec)

    records = []
    for i, frame in enumerate(container.decode(video=0)):
        if i % interval == 0:
            img = frame.to_ndarray(format="bgr24")
            records.append((img, i))

    return records, fps

def save_records(records, output_dir, output_csv, fps):
    """
    Saves filtered keyframes to disk and writes their metadata (path and timestamp) to a CSV file.

    Args:
        records (list): List of tuples (frame, frame_idx) to be saved.
        output_dir (str): Directory where the keyframe images will be stored.
        output_csv (str): Path to the output CSV file to store keyframe paths and timestamps.
        fps (float): Frames per second of the original video, used to calculate timestamps.

    Returns:
        pandas.DataFrame: DataFrame containing the saved keyframe paths and their corresponding timestamps.
    """
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for i, (frame, frame_idx) in enumerate(records):
        timestamp = _get_timestamp(frame_idx, fps)
        sanitized_timestamp = timestamp.replace(':', '-').replace('.', '-')
        frame_name = f"{sanitized_timestamp}.jpg"
        out_path = os.path.join(output_dir, frame_name)
        cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        rows.append([out_path, timestamp])

    df = pd.DataFrame(rows, columns=["keyframe", "timestamp"])
    df.to_csv(output_csv, index=False)
    return df

