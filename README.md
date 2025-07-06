# Visual Extraction Engine

An AI-powered engine for extracting and summarizing key visual content from video lectures using LLM agents, OCR, and graph-based workflows. This tool is optimized for downstream applications like educational summarization, content indexing, or tagging.

---

## 🚀 Features
- Intelligent keyframe selection based on visual and semantic importance
- Frame-level summarization using LangChain-compatible LLM agents
- OCR for slide and board content
- Modular graph-based frame processing pipeline
- Easy-to-configure paths and IO logic

---

## 🚧 Installation

```bash
git clone https://github.com/Video-Summerization-Project/Visual-Extraction-Engine.git
cd Visual-Extraction-Engine
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🔧 Usage

### 1. Prepare Inputs
- Create a `.env` file with required paths or configurations (see `config/paths.py` for reference)
- Add video files to a folder (e.g. `RawVideos/`)

### 2. Run Main Pipeline
```bash
python main.py --video_path RawVideos/example.mp4
```

### 3. Output
- Extracted keyframes: `outputs/keyframes/*.jpg`
- Summaries and tags: `outputs/final_output/results.json`
- CSV of keyframe metadata: `outputs/keyframes.csv`

---

## 📂 Project Structure

```
Visual-Extraction-Engine/
├── main.py                        # Entry script
├── requirements.txt              # Dependencies
├── config/                       # Path configs
│   └── paths.py
├── FrameProcessor/               # Main pipeline for frame processing
│   ├── main.py                   # Pipeline runner
│   ├── graph/                    # Graph-based frame processing workflow
│   │   ├── workflow.py
│   │   └── steps/
│   │       ├── describe_frame.py
│   │       ├── evaluate_importance.py
│   │       └── extract_features.py
│   ├── ocr/                      # OCR post-processing
│   │   └── describe_direct.py
│   ├── processor/                # Frame-wise processors
│   │   ├── multi_frame.py
│   │   └── single_frame.py
│   └── utils/                    # Utility functions
│       ├── image_utils.py
│       └── io_utils.py
├── KeyFrameSelection/           # Frame similarity + feature extractor
│   ├── FeatureExtraction.py
│   └── Similarties.py
├── llm/                         # LangChain-based LLM summarizer
│   └── model.py
├── notebooks/                   # Experiments & analysis
│   ├── Scene_detection.ipynb
│   └── Tags_Agent.ipynb
├── outputs/                     # Final outputs (frames, summaries)
│   ├── keyframes.csv
│   ├── final_output/results.json
│   └── keyframes/*.jpg
├── types_/                      # Shared data structures
│   └── state.py
```

> Note: `.env` file and `RawVideos/` folder are user-dependent and should be created manually.

---


## 📊 Notebooks
- `Scene_detection.ipynb`: Frame change detection logic
- `Tags_Agent.ipynb`: Experimental LLM-based tag generation

---

## 📡 Tech Stack
- Python 3.10+
- OpenCV, Tesseract OCR
- LangChain (LLM agent workflow)
- Custom graph processor architecture

---

## 🌐 License
MIT License. See `LICENSE` file for details.

---

## 🐝 Contributors
- [Amr Kahla](https://github.com/AmrKahla)
- [Team Members]

