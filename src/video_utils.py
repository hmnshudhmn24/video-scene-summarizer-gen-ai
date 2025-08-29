# src/video_utils.py
import os
import math
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple
import ffmpeg
from pytube import YouTube
from pydub import AudioSegment
from PIL import Image
import numpy as np
from tqdm import tqdm

TMP_DIR = Path(os.getenv("TMP_DIR", "./tmp"))
TMP_DIR.mkdir(parents=True, exist_ok=True)

def download_youtube_video(url: str, out_dir: str = None) -> str:
    """Download a YouTube video (progressive stream) and return local filepath."""
    out_dir = Path(out_dir or TMP_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    yt = YouTube(url)
    # pick progressive stream (audio+video) with reasonable resolution
    stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()
    if stream is None:
        # fallback to any mp4
        stream = yt.streams.filter(file_extension="mp4").first()
    out_path = stream.download(output_path=str(out_dir))
    return out_path

def extract_audio_from_video(video_path: str, out_audio_path: str = None, sample_rate: int = 16000) -> str:
    """Extract audio as WAV at sample_rate using ffmpeg/pydub. Returns path to WAV."""
    out_audio_path = out_audio_path or str(Path(TMP_DIR) / (Path(video_path).stem + "_audio.wav"))
    # use ffmpeg to convert
    (
        ffmpeg
        .input(video_path)
        .output(out_audio_path, ac=1, ar=sample_rate, format="wav", loglevel="error")
        .overwrite_output()
        .run()
    )
    return out_audio_path

def sample_frames(video_path: str, num_samples: int = 300, max_width: int = 640) -> List[Tuple[float, Image.Image]]:
    """
    Sample `num_samples` frames evenly across the video duration.
    Returns list of tuples (timestamp_seconds, PIL.Image).
    """
    # Get video duration via ffprobe
    try:
        probe = ffmpeg.probe(video_path)
        duration = float(probe["format"]["duration"])
    except Exception:
        # fallback to 0
        duration = 0.0

    if duration <= 0 or num_samples <= 0:
        return []

    timestamps = np.linspace(0, max(0.999 * duration, 0.001), num_samples)
    frames = []
    for t in tqdm(timestamps, desc="Sampling frames"):
        # extract one frame at timestamp t
        out_file = Path(TMP_DIR) / f"frame_{int(t*1000)}.jpg"
        try:
            (
                ffmpeg
                .input(video_path, ss=float(t))
                .output(str(out_file), vframes=1, format='image2', loglevel='error', s=f"{max_width}x?") # resize preserving aspect
                .overwrite_output()
                .run()
            )
            if out_file.exists():
                img = Image.open(out_file).convert("RGB")
                frames.append((float(t), img))
                out_file.unlink()  # remove temp file to save space
        except Exception:
            # skip failures
            pass
    return frames

def make_thumbnail(image: Image.Image, size=(320, 180)) -> Image.Image:
    img = image.copy()
    img.thumbnail(size, Image.LANCZOS)
    return img

    