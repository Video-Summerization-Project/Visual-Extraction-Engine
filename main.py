import warnings
warnings.filterwarnings("ignore")

from KeyFrameSelection.FeatureExtraction import process_video, save_records
from KeyFrameSelection.Similarties import hash_filter, clip_filter

keyframe_dir = 'outputs/keyframes'
csv_path = 'outputs/keyframes.csv'
video_path = 'RawVideos\Linear Regression - Hesham Asem (720p, h264).mp4'  # Change this to your video file path

def main():
    records, fps = process_video(video_path, interval_sec=10)

    min_frames = 10
    max_iterations = 10
    iteration = 0

    hash_threshold = 5
    ssim_threshold = 0.95
    clip_threshold = 0.90

    filtered = records

    while len(filtered) >= min_frames and iteration < max_iterations:
        filtered = hash_filter(
            filtered,
            hash_threshold=hash_threshold,
            ssim_threshold=ssim_threshold,
            ssim_compare_window=5
        )

        filtered = clip_filter(
            filtered,
            similarity_threshold=clip_threshold,
            compare_window=5
        )

        # Threshold tuning
        hash_threshold = max(1, hash_threshold - 1)
        ssim_threshold = max(0.5, ssim_threshold - 0.05)
        clip_threshold = min(0.99, clip_threshold + 0.03)

        iteration += 1
        print(f"Iter {iteration}: {len(filtered)} frames")

    save_records(filtered, keyframe_dir, csv_path, fps)

if __name__ == "__main__":
    import time
    start = time.time()
    main()
    end = time.time()
    print(f"Time to run is {end - start:.2f} sec")
    print("Keyframe selection process completed successfully.")
