from langgraph.graph import StateGraph, END
from types_.state import GraphState
from FrameProcessor.graph.steps.extract_features import extract_frame_features
from FrameProcessor.graph.steps.evaluate_importance import evaluate_importance
from FrameProcessor.graph.steps.describe_frame import describe_frame

def decide_next_step(state: GraphState) -> str:
    return state["next_step"]

def build_graph():
    workflow = StateGraph(GraphState)

    # Add steps
    workflow.add_node("extract_features", extract_frame_features)
    workflow.add_node("evaluate_importance", evaluate_importance)
    workflow.add_node("describe_frame", describe_frame)

    # Define transitions
    workflow.add_edge("extract_features", "evaluate_importance")
    workflow.add_conditional_edges(
        "evaluate_importance",
        decide_next_step,
        {
            "describe_frame": "describe_frame",
            END: END
        }
    )
    workflow.add_edge("describe_frame", END)

    workflow.set_entry_point("extract_features")

    return workflow.compile()

# Export the compiled graph
frame_processor = build_graph()
