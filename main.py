import warnings
warnings.filterwarnings("ignore")

from KeyFrameSelection.FeatureExtraction import process_video, save_records
from KeyFrameSelection.Similarties import hash_filter, clip_filter

keyframe_dir = 'outputs/keyframes'
csv_path = 'outputs/keyframes.csv'
video_path = 'RawVideos\Linear Regression - Hesham Asem (720p, h264).mp4' # Change this to your video file path

def main():
    # Process the video and extract keyframes
    records, fps = process_video(video_path, interval_sec=3)

    # Initial thresholds
    hash_threshold = 5
    ssim_threshold = 0.95
    clip_threshold = 0.90

    min_frames = 8
    max_iterations = 10
    iteration = 0

    filtered_records = records
    while len(filtered_records) > min_frames and iteration < max_iterations:
       # print(f"num of iteration {iteration+1}")
        filtered_records = hash_filter(
            filtered_records,
            hash_threshold=hash_threshold,
            ssim_threshold=ssim_threshold,
            ssim_compare_window=5
        )

        filtered_records = clip_filter(
            filtered_records,
            similarity_threshold=clip_threshold,
            compare_window=5
        )

        # Tighten thresholds
        hash_threshold = max(1, hash_threshold - 1)
        ssim_threshold = max(0.5, ssim_threshold - 0.05)
        clip_threshold = min(0.99, clip_threshold + 0.03)

        iteration += 1
        #print(f"Num of frames: {len(filtered_records)}")

    # Save the filtered keyframes and their records
    save_records(filtered_records, keyframe_dir, csv_path, fps)

if __name__ == "__main__":
    main()
    print("Keyframe selection process completed successfully.")
