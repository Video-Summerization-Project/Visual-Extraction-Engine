import warnings
warnings.filterwarnings("ignore")

from KeyFrameSelection.FeatureExtraction import process_video, save_records
from KeyFrameSelection.Similarties import hash_filter, clip_filter

keyframe_dir = 'outputs/keyframes'
csv_path = 'outputs/keyframes.csv'
video_path = 'https://drive.google.com/file/d/1B17NM3HS5tlN_gPzZjDlCBa2tKMTMwTN/view?usp=drive_link' # Change this to your video file path

def main():

    # Process the video to extract keyframes and their records
    records, fps = process_video(video_path, interval_sec=3)    # Process the video and extract keyframes
    filtered_records =hash_filter(records, hash_threshold=5, ssim_threshold=0.95, ssim_compare_window=5)    # Filter out similar keyframes based on hash and SSIM
    filtered_records = clip_filter(filtered_records, similarity_threshold=0.90, compare_window=5)   # Filter out similar keyframes based on clip similarity
    save_records(filtered_records, keyframe_dir, csv_path, fps)   # Save the filtered keyframes and their records to the specified directory and CSV file


if __name__ == "__main__":
    main()
    print("Keyframe selection process completed successfully.")
