import warnings
warnings.filterwarnings("ignore")
import os 
import shutil
import json


import time
from KeyFrameSelection.FeatureExtraction import process_video, save_records
from KeyFrameSelection.Similarties import hash_filter, clip_filter
from FrameProcessor.utils.io_utils import get_frames_from_folder, save_description_to_csv
from FrameProcessor.processor.multi_frame import process_frames
from config.paths import output_csv_file, output_json_file

# Input/output paths
keyframe_dir = 'outputs/keyframes'
csv_path = 'outputs/keyframes.csv'
video_path = 'RawVideos\Filters - Mohammad Ayed (720p, h264).mp4'  # Adjust as needed

def main():
    # Step 1: Extract raw keyframes from video
    records, fps = process_video(video_path, interval_sec=10)

    # Step 2: Filter keyframes
    min_frames = 10
    max_iterations = 20
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

    # Step 3: Save filtered keyframes
    save_records(filtered, keyframe_dir, csv_path, fps)
    print("Keyframe selection process completed successfully.")

    # Step 4: Process keyframes using FrameProcessor
    print("\n--- Frame Processing Started ---")
    frame_paths = get_frames_from_folder(keyframe_dir)
    results = process_frames(frame_paths)

    # Step 5: Show final summary
    important_frames = [r for r in results if r["importance"] == "important"]

    for result in important_frames:
        save_description_to_csv(result, output_csv_file)
    
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n--- Final Summary ---")
    print(f"Total frames processed: {len(results)}")
    print(f"Important frames: {len(important_frames)}")
    print(f"Output CSV: {output_csv_file}")
    print(f"Output JSON: {output_json_file}")

if __name__ == "__main__":
    start = time.time()

    if os.path.exists("outputs"):
        shutil.rmtree("outputs")
    os.makedirs("outputs/final_output", exist_ok=True)

    main()

    end = time.time()
    print(f"\nTotal time: {end - start:.2f} sec")
