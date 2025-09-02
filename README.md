# 🎬 Video Scene Summarizer (gen-ai)

Turn any YouTube video into a compact, visual timeline: automatic scene detection, thumbnails, and short, human-friendly summaries for each scene.

This project demonstrates a practical pipeline:
- 🎧 extract audio → transcribe with Whisper (local `faster-whisper` or OpenAI Whisper),
- 🎞️ sample video frames → embed using CLIP,
- 🧠 cluster embeddings to identify scenes,
- ✍️ generate short, natural-language summaries per scene using an LLM (OpenAI).



## ✨ Features

- Paste a YouTube URL or upload a local video file.
- Automatic audio extraction & transcription (Whisper).
- Sample frames evenly across the video and embed them with CLIP.
- Cluster frames into **N** scene groups and pick representative thumbnails.
- Use an LLM to create short headlines and summaries for each scene.
- Streamlit UI that shows thumbnails + timestamps + summaries.



## 📦 Project structure

```
video-scene-summarizer-gen-ai/
│── .env.example
│── requirements.txt
│── streamlit_app.py
└── src/
    ├── video_utils.py
    ├── transcribe.py
    ├── clip_utils.py
    └── summarizer.py
```



## 🛠️ Installation

> Recommended: use a virtual environment (conda/venv). GPU recommended for `faster-whisper` + CLIP + `torch`.

1. Clone repo
```bash
git clone https://github.com/yourname/video-scene-summarizer-gen-ai.git
cd video-scene-summarizer-gen-ai
```

2. Create & activate venv, install dependencies
```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and set keys
```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY (if you want LLM summaries or OpenAI Whisper)
```

---

## 🚀 Quickstart (run)

```bash
streamlit run streamlit_app.py
```

Open the Streamlit URL (typically `http://localhost:8501`).

Workflow:
1. Paste a YouTube link (or upload a short video).
2. Configure number of scenes, samples, and press **Process**.
3. Inspect the timeline: thumbnails + timestamps + LLM-generated summary text.



## ⚙️ Configuration (via `.env` or sidebar)

- `USE_OPENAI_WHISPER` — `true` to use OpenAI Whisper API instead of local transcription.
- `WHISPER_MODEL` — `large-v2` (faster-whisper) or other local model names.
- `CLIP_MODEL` — HF model ID for CLIP (e.g., `openai/clip-vit-base-patch32`).
- `NUM_FRAME_SAMPLES` — how many frames to sample from the video.
- `NUM_SCENES` — default number of scene clusters used by KMeans.
- `OPENAI_CHAT_MODEL` — model name used to produce summaries (e.g., `gpt-4o-mini`).



## 🧠 How it works (overview)

1. **Download or accept local video**: the app downloads the video via `pytube` or uses an uploaded file.
2. **Audio extraction**: ffmpeg extracts audio; saved as WAV (mono, 16 kHz).
3. **Transcription**: Whisper (local or OpenAI) produces time-stamped text segments.
4. **Frame sampling**: sample N frames evenly across video duration.
5. **CLIP embedding**: embed each sampled frame into vector space.
6. **Clustering**: KMeans clusters embeddings into `NUM_SCENES` clusters.
7. **Representative thumbnails**: for each cluster, pick frame(s) nearest to centroid.
8. **Segment transcripts**: associate transcript text (within ±window secs) to each cluster timestamp.
9. **LLM summarization**: call Chat model to generate a headline + short summary for each scene.
10. **UI rendering**: display thumbnails, timestamps, and summaries as a timeline.



## 🧩 Tips for quality

- Increase `NUM_FRAME_SAMPLES` for longer videos (more samples → better scene detection).
- Increase `NUM_SCENES` for finer-grained segmentation; lower for coarse scenes.
- Use GPU and prefer `faster-whisper` + `torch` with CUDA — dramatically faster for transcription and CLIP.
- For very long videos ( > 10 minutes ), sample fewer frames or run in a batch/offline mode to avoid memory/time issues.



## ⚠️ Limitations

- This is a prototype / research tool, not a production-grade scene detector. Real scene-change detection often uses shot-boundary detection + vision models.
- LLM summaries may hallucinate if transcript text is missing or poor; check the transcript segments to verify.
- YouTube downloads are subject to YouTube's terms of service — use responsibly and with permission where required.



## 🔧 Extensions & improvements

- Replace KMeans with shot-boundary detection + hierarchical clustering for better scenes.
- Use audio features (silence detection / energy) combined with visual embeddings to find scene cuts.
- Use an LLM to create a concise overall video summary and automatic chapter timestamps.
- Add multi-lingual transcript support and subtitle export (SRT/VTT).
- Add caching of embeddings and transcripts for re-runs.

