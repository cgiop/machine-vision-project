"""
ASL backend with alphabet, phrase, and gesture pipelines.
"""

import asyncio
import base64
import copy
import csv
import itertools
import json
import logging
import os
import sys
import time
from collections import Counter, deque
from urllib import error, parse, request

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel
from dotenv import load_dotenv
from twilio.rest import Client as TwilioClient

try:
    from gesture_heuristics import evaluate_gesture
except ModuleNotFoundError:
    from backend.gesture_heuristics import evaluate_gesture


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("asl")

load_dotenv(override=True)

BASE = os.path.dirname(__file__)
CNN_H5 = os.path.join(BASE, "cnn8grps_rad1_model.h5")
LSTM_H5 = os.path.join(BASE, "action.h5")
TFLITE = os.path.join(BASE, "kinivi_model", "keypoint_classifier", "keypoint_classifier.tflite")
KLABELS = os.path.join(BASE, "kinivi_model", "keypoint_classifier", "keypoint_classifier_label.csv")

PHRASE_LABELS = ["hello", "thanks", "iloveyou"]
SEQ_LEN = 30
MYMEMORY_URL = "https://api.mymemory.translated.net/get"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "").strip()
TWILIO_TO_NUMBER = os.getenv("TWILIO_TO_NUMBER", "").strip()
twilio_client = (
    TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN
    else None
)

with open(KLABELS, encoding="utf-8-sig") as f:
    KINIVI_LABELS = [row[0] for row in csv.reader(f)]

_cfg = {
    "threshold": 0.60,
    "alpha_hold_frames": 15,
}

AMBIGUOUS_LETTERS = {
    "A", "E", "M", "N", "S", "T",
    "C", "O",
    "G", "H",
    "P", "Q",
    "R", "U", "V",
}

log.info("Loading CNN A-Z model...")
alphabet_cnn = tf.keras.models.load_model(CNN_H5)
log.info("CNN loaded %s", alphabet_cnn.input_shape)

log.info("Loading LSTM phrase model...")
phrase_model = tf.keras.models.load_model(LSTM_H5)
log.info("LSTM loaded")

log.info("Loading TFLite gesture model...")
_interp = tf.lite.Interpreter(model_path=TFLITE, num_threads=2)
_interp.allocate_tensors()
_in_det = _interp.get_input_details()
_out_det = _interp.get_output_details()
log.info("TFLite loaded")

mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands


def _holistic_kp(res):
    pose = (
        np.array([[r.x, r.y, r.z, r.visibility] for r in res.pose_landmarks.landmark]).flatten()
        if res.pose_landmarks
        else np.zeros(33 * 4)
    )
    face = (
        np.array([[r.x, r.y, r.z] for r in res.face_landmarks.landmark]).flatten()
        if res.face_landmarks
        else np.zeros(468 * 3)
    )
    left_hand = (
        np.array([[r.x, r.y, r.z] for r in res.left_hand_landmarks.landmark]).flatten()
        if res.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    right_hand = (
        np.array([[r.x, r.y, r.z] for r in res.right_hand_landmarks.landmark]).flatten()
        if res.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    return np.concatenate([pose, face, left_hand, right_hand])


def _normalize_kinivi(lm_list):
    tmp = copy.deepcopy(lm_list)
    base_x, base_y = tmp[0]
    tmp = [[point[0] - base_x, point[1] - base_y] for point in tmp]
    flat = list(itertools.chain.from_iterable(tmp))
    max_abs = max(map(abs, flat)) or 1
    return [value / max_abs for value in flat]


def _tflite_run(feat):
    _interp.set_tensor(_in_det[0]["index"], np.array([feat], dtype=np.float32))
    _interp.invoke()
    probs = np.squeeze(_interp.get_tensor(_out_det[0]["index"]))
    idx = int(np.argmax(probs))
    return idx, float(probs[idx])


def _lm_px(bgr, lm_set):
    height, width = bgr.shape[:2]
    return [
        [min(int(landmark.x * width), width - 1), min(int(landmark.y * height), height - 1)]
        for landmark in lm_set.landmark
    ]


def _clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def _normalized_distance(point_a, point_b, width, height):
    dx = (point_a[0] - point_b[0]) / max(width, 1)
    dy = (point_a[1] - point_b[1]) / max(height, 1)
    return (dx * dx + dy * dy) ** 0.5


def _vision_metrics(frame, bbox, stability):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return {
            "vision_score": 0.0,
            "vision_feedback": ["Hand crop is empty. Reposition your hand."],
            "vision_metrics": {
                "brightness": 0.0,
                "sharpness": 0.0,
                "coverage": 0.0,
                "centering": 0.0,
                "stability": round(stability, 3),
            },
        }

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray) / 255.0)
    sharpness = _clamp(float(cv2.Laplacian(gray, cv2.CV_64F).var() / 180.0), 0.0, 1.0)
    coverage = _clamp(((x2 - x1) * (y2 - y1)) / max(width * height, 1), 0.0, 1.0)
    bbox_center = ((x1 + x2) / 2, (y1 + y2) / 2)
    frame_center = (width / 2, height / 2)
    center_distance = _normalized_distance(bbox_center, frame_center, width, height)
    centering = round(_clamp(1.0 - (center_distance / 0.45), 0.0, 1.0), 3)

    vision_score = round(
        (sharpness * 0.32)
        + (centering * 0.22)
        + (stability * 0.24)
        + (_clamp(1.0 - abs(brightness - 0.55) / 0.45, 0.0, 1.0) * 0.12)
        + (_clamp(min(coverage / 0.18, 1.0), 0.0, 1.0) * 0.10),
        3,
    )

    feedback = []
    if brightness < 0.22:
        feedback.append("Increase lighting on the hand.")
    elif brightness > 0.88:
        feedback.append("Reduce glare or move away from bright light.")

    if sharpness < 0.28:
        feedback.append("Hold still for a sharper frame.")

    if coverage < 0.05:
        feedback.append("Move your hand closer to the camera.")
    elif coverage > 0.38:
        feedback.append("Move your hand slightly back to fit the frame.")

    if centering < 0.55:
        feedback.append("Center your hand in the camera view.")

    if stability < 0.55:
        feedback.append("Keep the hand steadier before committing.")

    if not feedback:
        feedback.append("Frame quality looks good.")

    return {
        "vision_score": vision_score,
        "vision_feedback": feedback,
        "vision_metrics": {
            "brightness": round(brightness, 3),
            "sharpness": round(sharpness, 3),
            "coverage": round(coverage, 3),
            "centering": centering,
            "stability": round(stability, 3),
        },
    }


def _blank_result(mode, sentence="", seq_progress=0, hold_count=0):
    return {
        "mode": mode,
        "prediction": "",
        "confidence": 0.0,
        "stability": 0.0,
        "vision_score": 0.0,
        "vision_feedback": [],
        "vision_metrics": {},
        "boxes": [],
        "sentence": sentence,
        "hold_count": hold_count,
        "seq_progress": seq_progress,
    }


def _twilio_ready():
    return bool(twilio_client and TWILIO_FROM_NUMBER and TWILIO_TO_NUMBER)


def _send_open_hand_sms():
    if not _twilio_ready():
        return False, "Twilio is not configured"
    try:
        message = twilio_client.messages.create(
            from_=TWILIO_FROM_NUMBER,
            to=TWILIO_TO_NUMBER,
            body="Emergency Detected",
        )
        sid = getattr(message, "sid", "")
        return True, f"SMS sent{f' ({sid})' if sid else ''}"
    except Exception as exc:
        return False, f"Twilio send failed: {exc}"


class AlphabetSession:
    OFFSET = 29
    CANVAS = 400

    def __init__(self):
        self.hd = HandDetector(maxHands=1)
        self.hd2 = HandDetector(maxHands=1)
        self.sentence = ""
        self.prev_char = ""
        self.hold_count = 0
        self.hist = deque(maxlen=12)
        self.locked_prediction = ""
        self.motion_hist = deque(maxlen=8)
        self.last_bbox = None

    def _skeleton(self, pts, bbox):
        canvas = np.ones((self.CANVAS, self.CANVAS, 3), np.uint8) * 255
        if not pts or not bbox:
            return canvas

        _, _, box_w, box_h = bbox
        offset_x = ((self.CANVAS - box_w) // 2) - 15
        offset_y = ((self.CANVAS - box_h) // 2) - 15
        points = pts

        for idx in range(0, 4):
            cv2.line(canvas, (points[idx][0] + offset_x, points[idx][1] + offset_y), (points[idx + 1][0] + offset_x, points[idx + 1][1] + offset_y), (0, 255, 0), 3)
        for idx in range(5, 8):
            cv2.line(canvas, (points[idx][0] + offset_x, points[idx][1] + offset_y), (points[idx + 1][0] + offset_x, points[idx + 1][1] + offset_y), (0, 255, 0), 3)
        for idx in range(9, 12):
            cv2.line(canvas, (points[idx][0] + offset_x, points[idx][1] + offset_y), (points[idx + 1][0] + offset_x, points[idx + 1][1] + offset_y), (0, 255, 0), 3)
        for idx in range(13, 16):
            cv2.line(canvas, (points[idx][0] + offset_x, points[idx][1] + offset_y), (points[idx + 1][0] + offset_x, points[idx + 1][1] + offset_y), (0, 255, 0), 3)
        for idx in range(17, 20):
            cv2.line(canvas, (points[idx][0] + offset_x, points[idx][1] + offset_y), (points[idx + 1][0] + offset_x, points[idx + 1][1] + offset_y), (0, 255, 0), 3)

        for point_a, point_b in [(5, 9), (9, 13), (13, 17), (0, 5), (0, 17)]:
            cv2.line(canvas, (points[point_a][0] + offset_x, points[point_a][1] + offset_y), (points[point_b][0] + offset_x, points[point_b][1] + offset_y), (0, 255, 0), 3)
        for idx in range(21):
            cv2.circle(canvas, (points[idx][0] + offset_x, points[idx][1] + offset_y), 2, (0, 0, 255), 1)
        return canvas

    def _measure_stability(self, bbox):
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        area = max((x2 - x1) * (y2 - y1), 1)

        if self.last_bbox is None:
            self.last_bbox = (center, area)
            self.motion_hist.append(1.0)
            return 1.0

        (last_center_x, last_center_y), last_area = self.last_bbox
        shift = abs(center[0] - last_center_x) + abs(center[1] - last_center_y)
        area_delta = abs(area - last_area) / max(last_area, 1)
        motion_score = max(0.0, 1.0 - min(1.0, (shift / 42.0) + (area_delta * 0.65)))

        self.last_bbox = (center, area)
        self.motion_hist.append(motion_score)
        return round(sum(self.motion_hist) / len(self.motion_hist), 3)

    def process(self, bgr):
        frame = bgr
        frame_h, frame_w = frame.shape[:2]
        hold_frames = int(_cfg["alpha_hold_frames"])

        out = _blank_result(
            mode="alphabet",
            sentence=self.sentence,
            seq_progress=min(self.hold_count, hold_frames),
            hold_count=self.hold_count,
        )

        hands_result = self.hd.findHands(frame, draw=False, flipType=True)
        hands = hands_result[0] if isinstance(hands_result, (tuple, list)) and not isinstance(hands_result[0], dict) else hands_result
        if not hands:
            self.hist.clear()
            self.hold_count = 0
            self.prev_char = ""
            self.motion_hist.clear()
            self.last_bbox = None
            self.locked_prediction = ""
            return out

        hand = hands[0]
        x, y, box_w, box_h = hand["bbox"]
        y1 = max(0, y - self.OFFSET)
        y2 = min(frame_h, y + box_h + self.OFFSET)
        x1 = max(0, x - self.OFFSET)
        x2 = min(frame_w, x + box_w + self.OFFSET)
        out["boxes"] = [[x1, y1, x2, y2]]
        stability = self._measure_stability((x1, y1, x2, y2))
        out["stability"] = stability
        out.update(_vision_metrics(frame, (x1, y1, x2, y2), stability))

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return out

        roi_result = self.hd2.findHands(roi, draw=False, flipType=True)
        roi_hands = roi_result[0] if isinstance(roi_result, (tuple, list)) and not isinstance(roi_result[0], dict) else roi_result
        if not roi_hands:
            return out

        pts = roi_hands[0]["lmList"]
        bbox = roi_hands[0]["bbox"]
        canvas = self._skeleton(pts, bbox)

        probs = np.array(
            alphabet_cnn.predict(canvas.reshape(1, self.CANVAS, self.CANVAS, 3), verbose=0)[0],
            dtype="float32",
        )
        ch1 = int(np.argmax(probs))
        probs[ch1] = 0
        ch2 = int(np.argmax(probs))

        result = evaluate_gesture(ch1, ch2, pts)
        self.hist.append(result)

        predicted = ""
        conf = 0.0
        if len(self.hist) >= 6:
            best, count = Counter(self.hist).most_common(1)[0]
            ratio = count / len(self.hist)
            if ratio >= 0.5 and isinstance(best, str):
                predicted = best
                conf = round(ratio, 3)

        out["prediction"] = predicted
        out["confidence"] = conf

        if not predicted:
            self.hold_count = 0
            self.prev_char = ""
            self.locked_prediction = ""
            out["hold_count"] = 0
            out["seq_progress"] = 0
            return out

        if predicted == self.locked_prediction:
            out["sentence"] = self.sentence
            out["hold_count"] = 0
            out["seq_progress"] = 0
            return out

        if predicted == "Backspace":
            if self.prev_char != "Backspace":
                self.sentence = self.sentence[:-1]
            self.prev_char = "Backspace"
            self.hold_count = 0
            self.locked_prediction = "Backspace"
        elif predicted in ("next", " "):
            if self.prev_char != " ":
                self.sentence += " "
            self.prev_char = " "
            self.hold_count = 0
            self.locked_prediction = " "
        else:
            if predicted == self.prev_char:
                self.hold_count += 1
                if self.hold_count >= hold_frames:
                    self.sentence += predicted
                    self.hold_count = 0
                    self.locked_prediction = predicted
            else:
                self.prev_char = predicted
                self.hold_count = 0

        out["sentence"] = self.sentence
        out["hold_count"] = self.hold_count
        out["seq_progress"] = min(self.hold_count, hold_frames)
        return out

    def clear(self):
        self.sentence = ""
        self.prev_char = ""
        self.hold_count = 0
        self.hist.clear()
        self.locked_prediction = ""
        self.motion_hist.clear()
        self.last_bbox = None

    def set_sentence(self, value):
        self.sentence = value
        self.prev_char = ""
        self.hold_count = 0
        self.hist.clear()
        self.locked_prediction = ""
        self.motion_hist.clear()
        self.last_bbox = None

    def close(self):
        return None


class PhraseSession:
    def __init__(self):
        self.mp = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.seq = deque(maxlen=SEQ_LEN)
        self.votes = deque(maxlen=5)

    def process(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self.mp.process(rgb)
        rgb.flags.writeable = True
        height, width = bgr.shape[:2]

        boxes = []
        for hand in [res.left_hand_landmarks, res.right_hand_landmarks]:
            if hand:
                xs = [lm.x * width for lm in hand.landmark]
                ys = [lm.y * height for lm in hand.landmark]
                pad = 30
                boxes.append([
                    max(0, int(min(xs)) - pad),
                    max(0, int(min(ys)) - pad),
                    min(width, int(max(xs)) + pad),
                    min(height, int(max(ys)) + pad),
                ])

        self.seq.append(_holistic_kp(res))
        out = {
            "mode": "phrase",
            "prediction": "",
            "confidence": 0.0,
            "boxes": boxes,
            "seq_progress": len(self.seq),
            "sentence": "",
            "hold_count": 0,
        }

        if len(self.seq) == SEQ_LEN:
            probs = phrase_model.predict(np.expand_dims(np.array(self.seq), 0), verbose=0)[0]
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            self.votes.append(idx)
            if len(self.votes) == 5:
                best, _ = Counter(self.votes).most_common(1)[0]
                if conf >= _cfg["threshold"]:
                    out["prediction"] = PHRASE_LABELS[best]
                    out["confidence"] = round(conf, 3)
        return out

    def close(self):
        self.mp.close()


class GestureSession:
    def __init__(self):
        self.mp = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.5,
        )
        self.hist = deque(maxlen=10)
        self.open_streak = 0
        self.last_sms_at = 0.0
        self.sms_cooldown_seconds = 20.0
        self.last_sms_status = ""

    def _refresh_sms_status(self):
        if self.last_sms_at <= 0:
            return

        remaining = int(self.sms_cooldown_seconds - (time.monotonic() - self.last_sms_at))
        if remaining > 0:
            self.last_sms_status = f"Cooldown active ({remaining}s)"
        elif self.last_sms_status.startswith("Cooldown active"):
            self.last_sms_status = "SMS ready"

    def process(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self.mp.process(rgb)
        rgb.flags.writeable = True
        self._refresh_sms_status()
        out = {
            "mode": "gesture",
            "prediction": "",
            "confidence": 0.0,
            "boxes": [],
            "seq_progress": 10,
            "sentence": "",
            "hold_count": 0,
            "sms_ready": _twilio_ready(),
            "sms_status": self.last_sms_status,
        }

        if not res.multi_hand_landmarks:
            self.hist.clear()
            self.open_streak = 0
            out["sms_status"] = self.last_sms_status
            return out

        hand = res.multi_hand_landmarks[0]
        lm_list = _lm_px(bgr, hand)
        feat = _normalize_kinivi(lm_list)
        idx, _ = _tflite_run(feat)
        self.hist.append(idx)
        height, width = bgr.shape[:2]
        xs = [lm.x * width for lm in hand.landmark]
        ys = [lm.y * height for lm in hand.landmark]
        out["boxes"] = [[
            max(0, int(min(xs)) - 28),
            max(0, int(min(ys)) - 28),
            min(width, int(max(xs)) + 28),
            min(height, int(max(ys)) + 28),
        ]]

        if len(self.hist) >= 5:
            best, count = Counter(self.hist).most_common(1)[0]
            score = count / len(self.hist)
            if score >= 0.5:
                out["prediction"] = KINIVI_LABELS[best] if best < len(KINIVI_LABELS) else "?"
                out["confidence"] = round(score, 3)

        if out["prediction"] == "Open" and out["confidence"] >= 0.6:
            self.open_streak += 1
            if self.open_streak >= 3:
                now = time.monotonic()
                if now - self.last_sms_at >= self.sms_cooldown_seconds:
                    ok, status = _send_open_hand_sms()
                    self.last_sms_status = status
                    if ok:
                        self.last_sms_at = now
                else:
                    self._refresh_sms_status()
        else:
            self.open_streak = 0

        out["sms_status"] = self.last_sms_status
        return out

    def close(self):
        self.mp.close()


app = FastAPI(title="ASL Vision Backend", version="5.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status")
def status():
    return {
        "ok": True,
        "version": "5.2",
        "alpha_labels": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        "phrase_labels": PHRASE_LABELS,
        "gesture_labels": KINIVI_LABELS,
        "alpha_hold_frames": _cfg["alpha_hold_frames"],
        "groq_completion_enabled": bool(GROQ_API_KEY),
        "twilio_sms_enabled": _twilio_ready(),
        "config": _cfg,
    }


class Cfg(BaseModel):
    threshold: float | None = None
    alpha_hold_frames: int | None = None


class TranslateBody(BaseModel):
    text: str
    target_lang: str


class CompleteBody(BaseModel):
    text: str


def _normalize_completion_suggestions(base_text, raw_items):
    base_text = " ".join((base_text or "").strip().upper().split())
    if not base_text:
        return []

    prefix = base_text.split()[-1]
    normalized = []
    seen = set()

    for item in raw_items:
        candidate = " ".join(str(item or "").strip().upper().split())
        if not candidate:
            continue
        if not candidate.startswith(prefix):
            continue
        if not candidate.replace(" ", "").isalpha():
            continue
        if len(candidate) < len(prefix) + 1:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)
        if len(normalized) == 3:
            break

    return normalized


def _completion_context(text):
    clean_text = " ".join((text or "").strip().upper().split())
    if not clean_text:
        return "", ""
    parts = clean_text.split()
    return clean_text, parts[-1]


@app.post("/api/settings")
def update_cfg(body: Cfg):
    if body.threshold is not None:
        _cfg["threshold"] = body.threshold
    if body.alpha_hold_frames is not None:
        _cfg["alpha_hold_frames"] = body.alpha_hold_frames
    return {"ok": True, "config": _cfg}


@app.post("/api/translate")
def translate_text(body: TranslateBody):
    text = body.text.strip()
    target_lang = body.target_lang.strip().lower()
    if not text:
        return {"ok": False, "error": "Text is empty"}

    query = parse.urlencode({"q": text, "langpair": f"en|{target_lang}"})
    url = f"{MYMEMORY_URL}?{query}"
    try:
        with request.urlopen(url, timeout=12) as res:
            payload = json.loads(res.read().decode("utf-8"))
        translated = payload.get("responseData", {}).get("translatedText", "")
        if not translated:
            return {"ok": False, "error": "No translation returned"}
        return {"ok": True, "translated_text": translated}
    except error.URLError as exc:
        return {"ok": False, "error": f"Translation request failed: {exc.reason}"}
    except Exception as exc:
        return {"ok": False, "error": f"Translation failed: {exc}"}


@app.post("/api/complete")
def complete_text(body: CompleteBody):
    text = body.text.strip()
    if not text:
        return {"ok": False, "enabled": False, "error": "Text is empty"}

    if not groq_client:
        return {"ok": True, "enabled": False, "suggestions": []}

    clean_text, prefix = _completion_context(text)
    if len(prefix) < 2:
        return {"ok": True, "enabled": True, "suggestions": []}

    prompt = (
        "You are helping an ASL spelling app finish the current partially spelled English word. "
        "Return JSON only in the form {\"suggestions\":[\"WORD1\",\"WORD2\",\"WORD3\"]}. "
        "Rules: suggestions must be common real English words, all uppercase, no punctuation, "
        "no explanation, no invented words, and each suggestion must start with the exact unfinished word prefix. "
        f'Full text so far: "{clean_text}". '
        f'Current unfinished word prefix: "{prefix}"'
    )
    try:
        payload = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0,
            max_tokens=80,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return strict JSON only. Suggest three common English words for the current unfinished word only. "
                        "Do not output phrases, punctuation, or explanations."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = (payload.choices[0].message.content or "").strip()
        parsed = json.loads(content)
        suggestions = _normalize_completion_suggestions(prefix, parsed.get("suggestions", []))
        if not suggestions:
            return {"ok": True, "enabled": True, "suggestions": []}
        return {"ok": True, "enabled": True, "suggestions": suggestions}
    except Exception as exc:
        return {"ok": False, "enabled": True, "error": f"Groq completion failed: {exc}"}


_factories = {
    "alphabet": AlphabetSession,
    "phrase": PhraseSession,
    "gesture": GestureSession,
}


@app.websocket("/ws/feed")
async def ws_feed(ws: WebSocket, mode: str = Query(default="alphabet")):
    await ws.accept()
    session = _factories.get(mode, AlphabetSession)()
    log.info("WS connected mode=%s", mode)
    try:
        while True:
            raw = await ws.receive_text()

            if raw.startswith("{"):
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    message = None

                if isinstance(message, dict) and message.get("type") == "command":
                    if message.get("command") == "clear" and hasattr(session, "clear"):
                        session.clear()
                        await ws.send_text(json.dumps(_blank_result(mode=mode, sentence=getattr(session, "sentence", ""))))
                    elif message.get("command") == "set_sentence" and hasattr(session, "set_sentence"):
                        session.set_sentence(str(message.get("sentence") or ""))
                        await ws.send_text(json.dumps(_blank_result(mode=mode, sentence=getattr(session, "sentence", ""))))
                    continue

            if "," in raw:
                raw = raw.split(",", 1)[1]

            arr = np.frombuffer(base64.b64decode(raw), dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            result = await asyncio.get_running_loop().run_in_executor(None, session.process, frame)
            await ws.send_text(json.dumps(result))

    except WebSocketDisconnect:
        log.info("WS disconnected mode=%s", mode)
    except Exception as exc:
        log.exception("WS error mode=%s: %s", mode, exc)
    finally:
        session.close()
