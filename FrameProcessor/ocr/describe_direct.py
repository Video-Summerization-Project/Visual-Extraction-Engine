import os
import re
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from llm.model import model
from FrameProcessor.utils.image_utils import image_to_base64

def describe_frame_directly(frame_path: str) -> Dict[str, Any]:
    """Describe a frame directly without the state graph."""
    img_str, mime_type = image_to_base64(frame_path)
    if not img_str:
        return {
            "image_name": os.path.basename(frame_path),
            "extracted_text": "Failed to process image",
            "visual_description": "Error occurred during image processing",
            "error": "Image conversion failed"
        }

    prompt = f"""
        You are an expert in multilingual document understanding.

        Your task is to extract and analyze text and informative visual elements from the given image.

        Rules:
          - Analyze the provided image to extract all textual content.
          - If text is in Arabic, copy it in Arabic and provide an English translation in quotes immediately after the Arabic text.
          - If text is entirely in English, copy it as is.
          - If text is primarily Arabic with some English words, copy the Arabic text and place the English words in quotes within the Arabic text.
          - Additionally, identify any informative visual elements in the image that convey data or information.
          - This specifically includes elements such as charts, diagrams, text tables, histograms, flowcharts, illustrations, or other visual representations of data.
          - Do not describe the general image design, background, or purely decorative elements.
          - Translate the visual description to Arabic and remove English after translation.
          - Structure your output as follows, presenting the image information in a clear vertical format:

        Image Name: {os.path.basename(frame_path)}
        Extracted Text: [Copied text according to language rules, with English translations/quoted English words]
        Visual Description: [Detailed description of any informative visual elements present. State 'None' if no such visual elements are found.]
    """

    try:
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img_str}"}}
                ]
            )
        ]

        response = model.invoke(messages)
        output_text = response.content.strip()

        image_name_match = re.search(r'Image Name:\s*(.*?)\s*Extracted Text:', output_text, re.DOTALL) or \
                           re.search(r'اسم الصورة:\s*(.*?)\s*النص المستخرج:', output_text, re.DOTALL)

        extracted_text_match = re.search(r'Extracted Text:\s*(.*?)\s*Visual Description:', output_text, re.DOTALL) or \
                               re.search(r'النص المستخرج:\s*(.*?)\s*الوصف المرئي:', output_text, re.DOTALL)

        visual_description_match = re.search(r'Visual Description:\s*(.*)', output_text, re.DOTALL) or \
                                   re.search(r'الوصف المرئي:\s*(.*)', output_text, re.DOTALL)

        return {
            "image_name": image_name_match.group(1).strip() if image_name_match else os.path.basename(frame_path),
            "extracted_text": extracted_text_match.group(1).strip() if extracted_text_match else "No text found",
            "visual_description": visual_description_match.group(1).strip() if visual_description_match else "No visual description",
            "raw_output": output_text
        }

    except Exception as e:
        print(f"Error describing frame: {str(e)}")
        return {
            "image_name": os.path.basename(frame_path),
            "extracted_text": "Error occurred",
            "visual_description": f"Failed to analyze: {str(e)}",
            "raw_output": f"Error: {str(e)}"
        }
