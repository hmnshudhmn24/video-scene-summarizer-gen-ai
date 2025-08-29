# src/summarizer.py
import os
import openai
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

def summarize_segments_with_llm(segments: List[Dict], model: str = None, temperature: float = 0.2) -> List[Dict]:
    """
    segments: list of dicts { 'cluster': int, 'timestamp': float, 'transcript': '...'}
    Returns summaries: list of dicts {cluster, timestamp, summary, bullets}
    """
    model = model or OPENAI_CHAT_MODEL
    results = []
    for seg in segments:
        text = seg.get("transcript", "") or ""
        prompt = f"""You are an assistant that generates concise timeline summaries for video scenes.

Video excerpt transcript (may be short):
\"\"\"{text}\"\"\"

Please provide:
1) A very short headline (one-line) describing the key event in this excerpt.
2) A 1-2 sentence concise summary (what happened / important points).
3) Up to 3 short bullets with takeaways or actions (if applicable). Keep total output short.

Return JSON with keys: headline, summary, bullets.
"""
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a concise summarizer for video scenes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=200,
            )
            assistant = resp["choices"][0]["message"]["content"]
            results.append({"cluster": seg.get("cluster"), "timestamp": seg.get("timestamp"), "summary_text": assistant})
        except Exception as e:
            results.append({"cluster": seg.get("cluster"), "timestamp": seg.get("timestamp"), "summary_text": f"ERROR: {e}"})
    return results
