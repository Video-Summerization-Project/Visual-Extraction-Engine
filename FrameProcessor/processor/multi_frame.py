from typing import List, Dict, Any
from FrameProcessor.processor.single_frame import process_single_frame
from FrameProcessor.utils.io_utils import save_description_to_csv
from FrameProcessor.ocr.describe_direct import describe_frame_directly


def process_frames(frame_paths: List[str]) -> List[Dict[str, Any]]:
    """Process a set of video frames and evaluate their importance."""
    results = []
    important_frames_count = 0

    for i, frame_path in enumerate(frame_paths):
        print(f"Processing frame {i + 1}/{len(frame_paths)}: {frame_path}")

        try:
            result = process_single_frame(frame_path)
            results.append(result)

            if result["importance"] == "important":
                important_frames_count += 1
                print(f"   Important: {result['reason'][:50]}...")

                if "description" not in result or not result["description"]:
                    print(f"  Retrying description for important frame...")
                    try:
                        result["description"] = describe_frame_directly(frame_path)
                        print(f"  Description extracted successfully on retry")
                    except Exception as e:
                        print(f"  Retry failed: {str(e)}")

                if "description" in result and result["description"]:
                    save_description_to_csv(result)
            else:
                print(f"   Not important: {result['reason'][:50]}...")

        except Exception as e:
            print(f"  Error processing frame: {str(e)}")
            results.append({
                "frame": frame_path,
                "path": frame_path,
                "importance": "error",
                "reason": str(e)
            })

    print(f"\nFound {important_frames_count} important frames out of {len(frame_paths)}")
    return results
