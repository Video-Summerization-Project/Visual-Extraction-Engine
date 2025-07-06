import os
import csv
import glob
from typing import List, Dict
from config.paths import output_csv_file

def get_frames_from_folder(folder_path: str, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')) -> List[str]:
    """Extract paths of all image files from a specified folder."""
    frame_paths = []
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist")
        return frame_paths

    for ext in extensions:
        frame_paths.extend(glob.glob(os.path.join(folder_path, f'*{ext}')))

    frame_paths.sort()
    print(f"Found {len(frame_paths)} frames in folder {folder_path}")
    return frame_paths

_csv_initialized = False

def save_description_to_csv(result: Dict[str, any], output_csv_file: str= output_csv_file) -> bool:
    """Save important frame description to a CSV file (overwrite on first write)."""
    global _csv_initialized

    if "description" not in result or not result["description"]:
        print(f"  No description available for frame: {result.get('frame', '')}")
        return False

    description = result["description"]

    row = {
        "Image Name": description.get("image_name", os.path.basename(result.get("path", ""))),
        "Extracted Text": description.get("extracted_text", "No extracted text"),
        "Visual Description": description.get("visual_description", "No visual description")
    }

    print(f"  Extracted text ({len(row['Extracted Text'])} chars): {row['Extracted Text'][:50]}...")
    print(f"  Visual description ({len(row['Visual Description'])} chars): {row['Visual Description'][:50]}...")

    try:
        mode = 'w' if not _csv_initialized else 'a'
        with open(output_csv_file, mode, newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["Image Name", "Extracted Text", "Visual Description"])
            if not _csv_initialized:
                writer.writeheader()
                _csv_initialized = True
            writer.writerow(row)

        print(f"  Frame description saved to {output_csv_file}")
        return True
    except Exception as e:
        print(f" Error saving frame description: {str(e)}")
        return False
