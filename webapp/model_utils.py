"""
model_utils.py - Banana Analysis using Groq Vision API (Llama 4 Scout)
Replaces the overfitted BananaNet PyTorch model.
Capabilities:
  - Detect if the image contains a banana
  - Classify ripeness on a 1-5 scale (matching training data)
  - Estimate seed count based on visual analysis + training data statistics
"""

import os
import io
import json
import re
import base64
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # loads .env from project root
from PIL import Image

# ── Training data statistics for seed count estimation ──────────────────────
# Derived from clean_dataset.csv (32 samples)
SEED_STATS = {
    "min": 102,
    "max": 427,
    "mean": 261,
    "by_ripeness": {
        1: {"mean": 257, "min": 146, "max": 373},   # unripe
        2: {"mean": 281, "min": 147, "max": 421},   # slightly unripe
        3: {"mean": 267, "min": 173, "max": 375},   # semi-ripe
        4: {"mean": 295, "min": 102, "max": 392},   # ripe
        5: {"mean": 228, "min": 103, "max": 374},   # overripe
    }
}

RIPENESS_LABELS = {
    1: "Unripe 🟢",
    2: "Slightly Unripe 🟡",
    3: "Semi-Ripe 🟡",
    4: "Ripe ✅",
    5: "Overripe 🟤",
}

RIPENESS_DESCRIPTIONS = {
    1: "Green, firm, and starchy. Not ready to eat.",
    2: "Mostly green with slight yellowing. Nearly ready.",
    3: "Yellow with green tips. Good for cooking.",
    4: "Fully yellow, sweet, and ready to eat.",
    5: "Brown spots or fully brown. Very sweet, best for baking.",
}

# ── Groq API setup ───────────────────────────────────────────────────────────
# NOTE: Client is created lazily so a missing key produces a clean error
# response instead of crashing the Python process at import time (Vercel).
_client = None
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def _get_client() -> Groq:
    """Return (and cache) the Groq client, raising clearly if key is absent."""
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. "
                "Add it to your Vercel project's Environment Variables."
            )
        _client = Groq(api_key=api_key)
    return _client


def _encode_image_b64(image_path: str) -> str:
    """Read image, resize if needed, and return base64-encoded JPEG string."""
    img = Image.open(image_path).convert("RGB")
    # Limit to 1024px on longest side to stay within token limits
    img.thumbnail((1024, 1024), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def predict_from_image(image_path: str) -> dict:
    """
    Analyse a banana image using Groq Vision (Llama 4 Scout).

    Returns a dict with:
        is_banana      (bool)
        ripeness       (int 1-5, or None)
        ripeness_label (str)
        ripeness_desc  (str)
        seeds          (int estimated)
        seeds_range    (str e.g. "220 – 310")
        confidence     (str high/medium/low)
        banana_type    (str variety/type detected)
        notes          (str extra commentary)
        visual_clues   (str colour/texture observations)
        roast          (str savage one-liner roast)
        recommendations (list[str] 3 funny suggestions)
        error          (str only if something went wrong)
    """
    try:
        b64_image = _encode_image_b64(image_path)

        prompt = """You are PazhamAI — a savage banana expert and world-class Malayalam roast comedian. Analyze the image and respond ONLY with a valid JSON object. No markdown fences, no explanation, no extra text.

The JSON must follow this EXACT schema:
{
  "is_banana": true or false,
  "ripeness": integer 1-5 or null,
  "confidence": "high" or "medium" or "low",
  "banana_type": "e.g. Cavendish, Robusta, Nendran, Red Banana, Lady Finger, Ney Poovan, Baby Banana, or Unknown",
  "visual_clues": "brief colour/texture description of what you see",
  "notes": "any extra observation (freshness, defects, etc.)",
  "roast": "one savage, witty, single-sentence roast WRITTEN IN MALAYALAM (see rules below)"
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VARIETY-AWARE RIPENESS — CRITICAL:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NEVER judge ripeness by colour alone. Different varieties have different ripe colours:

  Robusta / Ney Poovan / Hill Banana (Kerala varieties):
    - Ripe (4) = skin stays GREEN or pale green — they NEVER turn yellow!
    - Signs of ripeness: slightly soft texture, rounded plump shape, no hard firmness
    - Overripe (5) = skin turns brown or black, mushy feel

  Nendran / Plantain (Kerala cooking banana — large, ribbed):
    - Ripe (4) = greenish-yellow to golden yellow, skin slightly thick
    - Overripe (5) = brown patches, very soft

  Cavendish (standard supermarket banana):
    - Unripe (1) = green, Ripe (4) = fully yellow, Overripe (5) = brown spots

  Red Banana / Cheruvaazha:
    - Ripe (4) = deep red/maroon/purple — NOT yellow
    - Unripe (1-2) = bright reddish-green, very firm
    - Overripe (5) = brownish-red with soft mushy spots

  Lady Finger / Baby Banana:
    - Ripe (4) = golden yellow, smaller and slender

Ripeness scale (apply AFTER identifying variety):
  1 = Unripe          (hard, starchy — not ready for THIS variety)
  2 = Slightly Unripe (close but not there yet)
  3 = Semi-Ripe       (borderline — edible but not peak)
  4 = Ripe            (perfect for this variety's expected appearance)
  5 = Overripe        (past peak — mushy, heavy browning/blackening)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROAST RULES — MUST BE IN MALAYALAM:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IF is_banana = false:
  - Roast the person for uploading a non-banana image to a BANANA AI. Mention what you actually see.
  - Write roast in Malayalam. Example: "ഇത് വാഴപ്പഴം അല്ല, പിന്നെ എന്തിനാ ഇവിടെ കൊണ്ടുവന്നത്?"
  - set ripeness = null, banana_type = null

IF is_banana = true — roast in Malayalam based on ripeness:
  Level 1 (Unripe): Roast for impatience.
    Example: "ഇത് ഇനിയും പഴുക്കണ്ടതാണ്, ക്ഷമ ഇല്ലേ നിനക്ക്?"
  Level 2 (Slightly Unripe): Almost-there roast.
    Example: "കൊഞ്ചം കൂടി കാത്തിരുന്നാൽ മതിയായിരുന്നു, ആർക്കും ക്ഷമയില്ല."
  Level 3 (Semi-Ripe): Middle-ground roast.
    Example: "ഇടത്തരം പഴം — ജീവിതം പോലെ, ഒന്നുമില്ല, ഒന്നും ശരിയുമല്ല."
  Level 4 (Ripe): Rare praise with a twist.
    Example: "ശരിയായി പഴുത്തു! ഇത് ഒരു അത്ഭുതം തന്നെ, അഭിനന്ദനങ്ങൾ."
  Level 5 (Overripe): Brutal neglect roast.
    Example: "ഈ പഴം നിന്നെ കാത്ത് കാത്ത് ചത്തു, നിനക്ക് ലജ്ജ തോന്നണം."
  Also reference banana TYPE if relevant, e.g., "ഇത് ഒരു റൊബസ്റ്റ ആണ്, പച്ചയാണ് എങ്കിലും നന്നായി പഴുത്തതാണ്."

ROAST must be ONE sentence only. Fluent, natural Malayalam. Funny and punchy.

Return ONLY the JSON. No markdown. No backticks."""

        response = _get_client().chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}"
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            temperature=0.75,  # higher temperature for creative, funny roasts
            max_tokens=1024,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if model wraps the JSON anyway
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        data = json.loads(raw)

        is_banana = bool(data.get("is_banana", False))

        if not is_banana:
            return {
                "is_banana": False,
                "ripeness": None,
                "ripeness_label": "N/A",
                "ripeness_desc": "No banana detected in the image.",
                "seeds": None,
                "seeds_range": "N/A",
                "confidence": data.get("confidence", "high"),
                "banana_type": None,
                "notes": data.get("notes", ""),
                "visual_clues": data.get("visual_clues", ""),
                "roast": data.get("roast", "That's definitely not a banana. Bold choice uploading it here."),
                "recommendations": [],
                "error": None,
            }

        ripeness = data.get("ripeness")
        if ripeness is not None:
            ripeness = max(1, min(5, int(ripeness)))

        seeds, seeds_range = _estimate_seeds(ripeness)

        recs = data.get("recommendations", [])
        if not isinstance(recs, list):
            recs = []

        return {
            "is_banana": True,
            "ripeness": ripeness,
            "ripeness_label": RIPENESS_LABELS.get(ripeness, "Unknown"),
            "ripeness_desc": RIPENESS_DESCRIPTIONS.get(ripeness, ""),
            "seeds": seeds,
            "seeds_range": seeds_range,
            "confidence": data.get("confidence", "medium"),
            "banana_type": data.get("banana_type", "Unknown"),
            "notes": data.get("notes", ""),
            "visual_clues": data.get("visual_clues", ""),
            "roast": data.get("roast", ""),
            "recommendations": recs,
            "error": None,
        }

    except json.JSONDecodeError as e:
        return _error_result(f"Could not parse model response as JSON: {e}. Raw: {raw[:200]}")
    except Exception as e:
        return _error_result(str(e))


def _estimate_seeds(ripeness) -> tuple:
    """Return (mean_estimate, 'min – max') based on ripeness level from training data."""
    if ripeness and ripeness in SEED_STATS["by_ripeness"]:
        stats = SEED_STATS["by_ripeness"][ripeness]
        return stats["mean"], f"{stats['min']} – {stats['max']}"
    return SEED_STATS["mean"], f"{SEED_STATS['min']} – {SEED_STATS['max']}"


def _error_result(msg: str) -> dict:
    return {
        "is_banana": None,
        "ripeness": None,
        "ripeness_label": "Error",
        "ripeness_desc": "",
        "seeds": None,
        "seeds_range": "N/A",
        "confidence": "low",
        "banana_type": None,
        "notes": "",
        "visual_clues": "",
        "roast": "",
        "recommendations": [],
        "error": msg,
    }
