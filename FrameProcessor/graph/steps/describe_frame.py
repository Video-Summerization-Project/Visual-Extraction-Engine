# graph/steps/describe_frame.py

import os
import re
from langchain_core.messages import HumanMessage
from llm.model import model
from langgraph.graph import END
from types_.state import GraphState

def describe_frame(state: GraphState) -> GraphState:
    """Extract detailed description and OCR from important frame."""
    frame_path = state["frame_path"]

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
- Translate the visual description to Arabic if needed.

Structure your output in this format:

Image Name: {os.path.basename(frame_path)}
Extracted Text: [copied text with translations]
Visual Description: [description in Arabic of any informative visuals]
"""

    try:
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{state['frame_data']['base64_image']}"}}
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

        state["description"] = {
            "image_name": image_name_match.group(1).strip() if image_name_match else os.path.basename(frame_path),
            "extracted_text": extracted_text_match.group(1).strip() if extracted_text_match else "No text found",
            "visual_description": visual_description_match.group(1).strip() if visual_description_match else "No visual description",
            "raw_output": output_text
        }

    except Exception as e:
        print(f"Error describing frame: {str(e)}")
        state["description"] = {
            "image_name": os.path.basename(frame_path),
            "extracted_text": "Error processing text",
            "visual_description": "Error generating description",
            "error": str(e)
        }

    state["next_step"] = END
    return state
