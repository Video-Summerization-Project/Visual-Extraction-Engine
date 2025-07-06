import os
from typing import Dict, Any
from FrameProcessor.graph.workflow import frame_processor
from FrameProcessor.ocr.describe_direct import describe_frame_directly

def process_single_frame(frame_path: str) -> Dict[str, Any]:
    """Process a single video frame: classify and if important, describe it."""
    initial_state = {
        "frame_path": frame_path,
        "frame_data": {},
        "frame_features": {},
        "importance": "not_important",
        "reason": "",
        "description": {},
        "next_step": "extract_features"
    }

    try:
        # Execute the state graph
        result = frame_processor.invoke(initial_state)

        output = {
            "frame": os.path.basename(frame_path),
            "path": frame_path,
            "importance": result["importance"],
            "reason": result["reason"],
        }

        # If frame is important, ensure description is present
        if result["importance"] == "important":
            if "description" in result and isinstance(result["description"], dict) and result["description"]:
                output["description"] = result["description"]
            else:
                print(f"  Warning: No description extracted for important frame: {os.path.basename(frame_path)}")
                try:
                    output["description"] = describe_frame_directly(frame_path)
                    print(f"   Description extracted successfully on second attempt")
                except Exception as e:
                    print(f"   Failed to extract description: {str(e)}")
                    output["description"] = {
                        "image_name": os.path.basename(frame_path),
                        "extracted_text": "Failed to extract text",
                        "visual_description": "Failed to extract visual description",
                        "error": str(e)
                    }

        return output

    except Exception as e:
        print(f"  Error processing frame: {str(e)}")
        return {
            "frame": os.path.basename(frame_path),
            "path": frame_path,
            "importance": "error",
            "reason": f"Processing error: {str(e)}",
            "error": str(e)
        }
