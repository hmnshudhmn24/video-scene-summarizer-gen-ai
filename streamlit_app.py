# streamlit_app.py
import os
import math
import time
import tempfile
from pathlib import Path
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

from src.video_utils import download_youtube_video, extract_audio_from_video, sample_frames, make_thumbnail, save_thumbnail, cleanup_tmp, TMP_DIR
from src.transcribe import transcribe
from src.clip_utils import embed_frames, cluster_embeddings, pick_representative_frames
from src.summarizer import summarize_segments_with_llm

# Config
NUM_FRAME_SAMPLES = int(os.getenv("NUM_FRAME_SAMPLES", "300"))
NUM_SCENES = int(os.getenv("NUM_SCENES", "8"))
TOP_K = int(os.getenv("TOP_K_FRAMES_PER_CLUSTER", "1"))

st.set_page_config(page_title="Video Scene Summarizer (gen-ai)", page_icon="🎬", layout="wide")
st.title("🎬 Video Scene Summarizer — YouTube → Timeline + Thumbnails + Short Summaries")
st.markdown("Paste a YouTube URL, then press **Process**. The app will download the video (or use a short clip), extract audio & frames, transcribe, cluster frames into scenes, and produce a timeline summary with thumbnails.")

with st.sidebar:
    st.header("Settings")
    num_samples = st.number_input("Frame samples", min_value=50, max_value=2000, value=NUM_FRAME_SAMPLES, step=50)
    n_scenes = st.slider("Number of scenes (clusters)", 2, 20, NUM_SCENES)
    top_k = st.number_input("Thumbnails per scene", 1, 3, TOP_K)
    use_local_transcribe = st.checkbox("Use local whisper (faster-whisper) if available", value=True)
    cleanup = st.checkbox("Clean temporary files after run", value=False)

url = st.text_input("YouTube URL (or local mp4 path):")
uploaded_file = st.file_uploader("Or upload a local mp4 file", type=["mp4", "mov", "mkv"])

if st.button("🚀 Process Video"):
    if not url and uploaded_file is None:
        st.error("Provide a YouTube URL or upload a local video file.")
    else:
        st.info("This may take some minutes (depending on video length & model speed).")
        # Prepare video_path
        if uploaded_file is not None:
            tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmpfile.write(uploaded_file.read())
            tmpfile.flush()
            video_path = tmpfile.name
        else:
            with st.spinner("Downloading video..."):
                try:
                    video_path = download_youtube_video(url)
                except Exception as e:
                    st.error(f"Failed to download: {e}")
                    video_path = None
        if not video_path:
            st.stop()

        st.success(f"Video ready: {video_path}")
        # Extract audio
        with st.spinner("Extracting audio..."):
            audio_path = extract_audio_from_video(video_path)
        st.write(f"Audio extracted: {audio_path}")

        # Transcribe
        with st.spinner("Transcribing audio (Whisper)..."):
            try:
                segments = transcribe(audio_path)  # list of {start,end,text}
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                segments = [{"start": 0.0, "end": None, "text": ""}]
        st.write(f"Got {len(segments)} transcript segments.")

        # Sample frames
        with st.spinner("Sampling frames from video..."):
            frames = sample_frames(video_path, num_samples)
        st.write(f"Sampled {len(frames)} frames.")

        # Embed frames
        with st.spinner("Embedding frames with CLIP..."):
            try:
                embeddings, timestamps = embed_frames(frames, batch_size=32)
            except Exception as e:
                st.error(f"CLIP embedding failed: {e}")
                embeddings, timestamps = None, None

        if embeddings is None or len(embeddings) == 0:
            st.error("No embeddings extracted; aborting.")
            st.stop()

        # Cluster embeddings
        with st.spinner("Clustering frames into scenes..."):
            labels = cluster_embeddings(embeddings, n_clusters=n_scenes)
        st.write(f"Found {len(set(labels))} scene clusters.")

        # Pick representative frames
        reps = pick_representative_frames(embeddings, frames, labels, top_k)
        st.write(f"Picked {len(reps)} representative thumbnails.")

        # Map clusters -> transcript text (concatenate transcript segments near the timestamps)
        # For each representative timestamp, gather transcript within a window (e.g., +/- 7 seconds)
        window = 7.0
        cluster_transcripts = []
        for r in reps:
            ts = r["timestamp"]
            # collect text segments whose midpoint is within window
            texts = []
            for seg in segments:
                seg_start = float(seg.get("start", 0.0) or 0.0)
                seg_end = seg.get("end", None)
                if seg_end is None:
                    seg_end = seg_start + 5.0  # fallback
                seg_mid = (seg_start + seg_end) / 2.0
                if abs(seg_mid - ts) <= window:
                    texts.append(seg.get("text", ""))
            transcript_concat = " ".join(texts).strip()
            cluster_transcripts.append({"cluster": r["cluster"], "timestamp": ts, "transcript": transcript_concat})

        # Summarize each cluster with LLM
        with st.spinner("Summarizing scenes with LLM..."):
            try:
                summaries = summarize_segments_with_llm(cluster_transcripts)
            except Exception as e:
                st.error(f"Summarization failed: {e}")
                summaries = [{"cluster": c["cluster"], "timestamp": c["timestamp"], "summary_text": c["transcript"]} for c in cluster_transcripts]

        # Display timeline: one row per representative cluster
        st.markdown("## ⏱️ Timeline Summary")
        rows = []
        for rep, summ in zip(reps, summaries):
            # save thumbnail to tmp for display
            thumb = make_thumbnail(rep["image"], size=(360, 200))
            thumb_path = Path(TMP_DIR) / f"thumb_cluster_{rep['cluster']}.jpg"
            save_thumbnail(thumb, thumb_path)
            col1, col2 = st.columns([1,3])
            with col1:
                st.image(thumb_path, use_column_width=True)
                st.caption(f"t = {rep['timestamp']:.1f}s")
            with col2:
                st.markdown(f"**Scene {rep['cluster']} — t={rep['timestamp']:.1f}s**")
                st.markdown(summ.get("summary_text", ""))
        if cleanup:
            cleanup_tmp()
            st.info("Temporary files cleaned up.")
