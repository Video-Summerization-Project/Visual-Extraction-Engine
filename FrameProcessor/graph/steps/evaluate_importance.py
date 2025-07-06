import json
import re
from langchain_core.messages import HumanMessage, SystemMessage
from llm.model import model
from langgraph.graph import END
from types_.state import GraphState

def evaluate_importance(state: GraphState) -> GraphState:
    """Use LLM to determine whether the frame is important."""

    if state["frame_features"].get("dark_ratio", 0) > 0.9:
        state["importance"] = "not_important"
        state["reason"] = "Frame is mostly black (over 90%)"
        state["next_step"] = END
        return state

    if "error" in state["frame_features"]:
        state["importance"] = "not_important"
        state["reason"] = f"Could not properly analyze frame: {state['frame_features']['error']}"
        state["next_step"] = END
        return state

    try:
        messages = [
            SystemMessage(content="""You are an expert in video summarization. Your task is to evaluate the importance of a video frame for inclusion in a video summary.

Evaluate the frame and classify it as either "important" or "not_important" based on the following criteria:

Important frames:
- Contain essential information for the video
- Show important events or scene changes
- Contain important text or visual information
- Represent key moments in the video

Unimportant frames:
- Black or single-color frames
- Regular portrait shots unrelated to video content
- Transitional or blurry frames
- Frames very similar to previous ones

Return a JSON containing:
{
  "importance": "important" or "not_important",
  "reason": "reason for your classification"
}
"""),
            HumanMessage(
                content=[
                    {"type": "text", "text": "Evaluate the importance of this video frame."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{state['frame_data']['base64_image']}"}}
                ]
            )
        ]

        response = model.invoke(messages)

        try:
            json_match = re.search(r'({.*})', response.content.replace('\n', ' '))
            if json_match:
                result = json.loads(json_match.group(1))
                state["importance"] = result.get("importance", "not_important")
                state["reason"] = result.get("reason", "No reason provided")
            else:
                state["importance"] = "important" if "important" in response.content.lower() else "not_important"
                state["reason"] = response.content

        except Exception as e:
            print(f"Error parsing importance response: {str(e)}")
            state["importance"] = "not_important"
            state["reason"] = f"Error processing response: {str(e)}"

    except Exception as e:
        print(f"Error evaluating importance: {str(e)}")
        state["importance"] = "not_important"
        state["reason"] = f"Failed to evaluate: {str(e)}"

    state["next_step"] = "describe_frame" if state["importance"] == "important" else END
    return state
