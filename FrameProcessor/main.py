from processor.multi_frame import process_frames
from utils.io_utils import get_frames_from_folder
from config.paths import output_json_file, output_csv_file
import os
import json

def main(frames_folder: str):
    """Main function to process frames in a folder"""
    # Ensure output folder exists
    os.makedirs("output", exist_ok=True)

    # Get list of all frames in folder
    frame_paths = get_frames_from_folder(frames_folder)

    if not frame_paths:
        print("No frames found for processing!")
        return

    # Process frames
    results = process_frames(frame_paths)

    # Save classification results to JSON file
    with open(output_json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nClassification results saved to {output_json_file}")

    # Display summary
    important_frames = [r for r in results if r["importance"] == "important"]
    print("\n--- Results Summary ---")
    print(f"Total frames: {len(results)}")
    print(f"Important frames: {len(important_frames)}")
    print(f"Unimportant frames: {len(results) - len(important_frames)}")
    print(f"Descriptions saved to: {output_csv_file}")
    print(f"Raw results saved to: {output_json_file}")

    # Print important frames with reasons
    if important_frames:
        print("\n--- Important Frames ---")
        for i, frame in enumerate(important_frames):
            print(f"{i+1}. {frame['frame']}: {frame['reason'][:100]}...")

# 🔁 Hardcoded path here
if __name__ == "__main__":
    frames_folder_path = "data/frames"  # ✅ Change this path as needed
    main(frames_folder_path)
