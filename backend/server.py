"""
ASL Alphabet Backend — alphabet-first, letter detection is THE feature.

Default mode: alphabet (A-Z via CNN + cvzone + heuristics)
Also available: phrase (LSTM hello/thanks/iloveyou), gesture (TFLite kinivi)
"""

import asyncio, base64, copy, csv, itertools, json, logging, math, os, sys
from collections import Counter, deque

import cv2, numpy as np
import mediapipe as mp
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from gesture_heuristics import evaluate_gesture          # run directly
except ModuleNotFoundError:
    from backend.gesture_heuristics import evaluate_gesture  # run as package

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("asl")

BASE     = os.path.dirname(__file__)
CNN_H5   = os.path.join(BASE, "cnn8grps_rad1_model.h5")
LSTM_H5  = os.path.join(BASE, "action.h5")
TFLITE   = os.path.join(BASE, "kinivi_model", "keypoint_classifier", "keypoint_classifier.tflite")
KLABELS  = os.path.join(BASE, "kinivi_model", "keypoint_classifier", "keypoint_classifier_label.csv")

PHRASE_LABELS = ["hello", "thanks", "iloveyou"]
SEQ_LEN       = 30

with open(KLABELS, encoding="utf-8-sig") as f:
    KINIVI_LABELS = [r[0] for r in csv.reader(f)]

_cfg = {"threshold": 0.60, "alpha_hold_frames": 15}

# ── load models ───────────────────────────────────────────────────────────────
log.info("Loading CNN A-Z model …")
alphabet_cnn = tf.keras.models.load_model(CNN_H5)
log.info("CNN loaded — %s", alphabet_cnn.input_shape)

log.info("Loading LSTM phrase model …")
phrase_model = tf.keras.models.load_model(LSTM_H5)
log.info("LSTM loaded")

log.info("Loading TFLite gesture model …")
_interp = tf.lite.Interpreter(model_path=TFLITE, num_threads=2)
_interp.allocate_tensors()
_in_det  = _interp.get_input_details()
_out_det = _interp.get_output_details()
log.info("TFLite loaded")

mp_holistic = mp.solutions.holistic
mp_hands    = mp.solutions.hands


# ── helpers ───────────────────────────────────────────────────────────────────
def _holistic_kp(res):
    pose = (np.array([[r.x,r.y,r.z,r.visibility] for r in res.pose_landmarks.landmark]).flatten()
            if res.pose_landmarks else np.zeros(33*4))
    face = (np.array([[r.x,r.y,r.z] for r in res.face_landmarks.landmark]).flatten()
            if res.face_landmarks else np.zeros(468*3))
    lh   = (np.array([[r.x,r.y,r.z] for r in res.left_hand_landmarks.landmark]).flatten()
            if res.left_hand_landmarks else np.zeros(21*3))
    rh   = (np.array([[r.x,r.y,r.z] for r in res.right_hand_landmarks.landmark]).flatten()
            if res.right_hand_landmarks else np.zeros(21*3))
    return np.concatenate([pose, face, lh, rh])

def _normalize_kinivi(lm_list):
    tmp = copy.deepcopy(lm_list)
    bx, by = tmp[0]
    tmp  = [[p[0]-bx, p[1]-by] for p in tmp]
    flat = list(itertools.chain.from_iterable(tmp))
    mx   = max(map(abs, flat)) or 1
    return [v/mx for v in flat]

def _tflite_run(feat):
    _interp.set_tensor(_in_det[0]['index'], np.array([feat], dtype=np.float32))
    _interp.invoke()
    probs = np.squeeze(_interp.get_tensor(_out_det[0]['index']))
    idx   = int(np.argmax(probs))
    return idx, float(probs[idx])

def _lm_px(bgr, lm_set):
    h, w = bgr.shape[:2]
    return [[min(int(l.x*w), w-1), min(int(l.y*h), h-1)] for l in lm_set.landmark]


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION: ALPHABET  (main engine)
# ══════════════════════════════════════════════════════════════════════════════
class AlphabetSession:
    """
    CNN A-Z via cvzone HandDetector + skeleton canvas + gesture_heuristics.
    Exactly the Sign-Language-To-Text pipeline, productionised.
    Hold any letter steady for `alpha_hold_frames` frames → auto-commit.
    """
    OFFSET = 29
    CANVAS = 400

    def __init__(self):
        self.hd        = HandDetector(maxHands=1)
        self.hd2       = HandDetector(maxHands=1)
        self.sentence  = ""
        self.prev_char = ""
        self.hold_count = 0
        self.hist      = deque(maxlen=12)

    def _skeleton(self, pts, bbox):
        w = np.ones((self.CANVAS, self.CANVAS, 3), np.uint8) * 255
        if not pts or not bbox:
            return w
        _, _, bw, bh = bbox
        ox = ((self.CANVAS - bw) // 2) - 15
        oy = ((self.CANVAS - bh) // 2) - 15
        P  = pts
        for t in range(0, 4): cv2.line(w,(P[t][0]+ox,P[t][1]+oy),(P[t+1][0]+ox,P[t+1][1]+oy),(0,255,0),3)
        for t in range(5, 8): cv2.line(w,(P[t][0]+ox,P[t][1]+oy),(P[t+1][0]+ox,P[t+1][1]+oy),(0,255,0),3)
        for t in range(9,12): cv2.line(w,(P[t][0]+ox,P[t][1]+oy),(P[t+1][0]+ox,P[t+1][1]+oy),(0,255,0),3)
        for t in range(13,16):cv2.line(w,(P[t][0]+ox,P[t][1]+oy),(P[t+1][0]+ox,P[t+1][1]+oy),(0,255,0),3)
        for t in range(17,20):cv2.line(w,(P[t][0]+ox,P[t][1]+oy),(P[t+1][0]+ox,P[t+1][1]+oy),(0,255,0),3)
        pairs = [(5,9),(9,13),(13,17),(0,5),(0,17)]
        for a,b in pairs: cv2.line(w,(P[a][0]+ox,P[a][1]+oy),(P[b][0]+ox,P[b][1]+oy),(0,255,0),3)
        for i in range(21): cv2.circle(w,(P[i][0]+ox,P[i][1]+oy),2,(0,0,255),1)
        return w

    def process(self, bgr):
        # No flip — browser sends raw unmirrored frames, CNN was trained on raw orientation
        # Result: right-hand signs work correctly (dominant hand for most users)
        frame  = bgr
        fh, fw = frame.shape[:2]
        hold_frames = int(_cfg["alpha_hold_frames"])

        out = {"mode":"alphabet","prediction":"","confidence":0.0,
               "boxes":[],"sentence":self.sentence,
               "hold_count":self.hold_count,"seq_progress":0}

        # cvzone findHands returns (hands_list, img) tuple in newer versions
        _result = self.hd.findHands(frame, draw=False, flipType=True)
        hands   = _result[0] if isinstance(_result, (tuple, list)) and not isinstance(_result[0], dict) else _result
        if not hands:
            self.hist.clear(); self.hold_count = 0
            return out

        hand = hands[0]
        x, y, bw, bh = hand['bbox']
        y1 = max(0, y  - self.OFFSET); y2 = min(fh, y  + bh + self.OFFSET)
        x1 = max(0, x  - self.OFFSET); x2 = min(fw, x  + bw + self.OFFSET)
        out["boxes"] = [[x1, y1, x2, y2]]

        roi   = frame[y1:y2, x1:x2]
        if roi.size == 0: return out

        _result2 = self.hd2.findHands(roi, draw=False, flipType=True)
        handz    = _result2[0] if isinstance(_result2, (tuple, list)) and not isinstance(_result2[0], dict) else _result2
        if not handz: return out

        pts  = handz[0]['lmList']
        bbox = handz[0]['bbox']
        canvas = self._skeleton(pts, bbox)

        probs = np.array(alphabet_cnn.predict(
            canvas.reshape(1, self.CANVAS, self.CANVAS, 3), verbose=0)[0], dtype='float32')
        ch1 = int(np.argmax(probs)); probs[ch1] = 0; ch2 = int(np.argmax(probs))

        result = evaluate_gesture(ch1, ch2, pts)
        self.hist.append(result)

        predicted = ""
        conf      = 0.0
        if len(self.hist) >= 6:
            best, cnt = Counter(self.hist).most_common(1)[0]
            ratio = cnt / len(self.hist)
            if ratio >= 0.5 and isinstance(best, str):
                predicted = best
                conf      = round(ratio, 3)

        out["prediction"] = predicted
        out["confidence"] = conf

        if predicted:
            hf = hold_frames
            if predicted == "Backspace":
                if self.prev_char != "Backspace":
                    self.sentence = self.sentence[:-1]
                self.prev_char  = "Backspace"
                self.hold_count = 0
            elif predicted in ("next", " "):
                if self.prev_char != " ":
                    self.sentence += " "
                self.prev_char  = " "
                self.hold_count = 0
            else:
                if predicted == self.prev_char:
                    self.hold_count += 1
                    if self.hold_count >= hf:
                        self.sentence  += predicted
                        self.hold_count = 0
                else:
                    self.prev_char  = predicted
                    self.hold_count = 0

        out["sentence"]     = self.sentence
        out["hold_count"]   = self.hold_count
        out["seq_progress"] = min(self.hold_count, hold_frames)
        return out

    def clear(self):
        self.sentence = ""; self.prev_char = ""; self.hold_count = 0

    def close(self): pass


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION: PHRASE  (secondary)
# ══════════════════════════════════════════════════════════════════════════════
class PhraseSession:
    def __init__(self):
        self.mp  = mp_holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.seq = deque(maxlen=SEQ_LEN)
        self.votes = deque(maxlen=5)

    def process(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); rgb.flags.writeable = False
        res = self.mp.process(rgb); rgb.flags.writeable = True
        h, w = bgr.shape[:2]

        boxes = []
        for hand in [res.left_hand_landmarks, res.right_hand_landmarks]:
            if hand:
                xs = [lm.x*w for lm in hand.landmark]; ys = [lm.y*h for lm in hand.landmark]
                pad = 30
                boxes.append([max(0,int(min(xs))-pad),max(0,int(min(ys))-pad),
                               min(w,int(max(xs))+pad),min(h,int(max(ys))+pad)])

        self.seq.append(_holistic_kp(res))
        out = {"mode":"phrase","prediction":"","confidence":0.0,"boxes":boxes,
               "seq_progress":len(self.seq),"sentence":"","hold_count":0}

        if len(self.seq) == SEQ_LEN:
            probs = phrase_model.predict(np.expand_dims(np.array(self.seq),0), verbose=0)[0]
            idx   = int(np.argmax(probs)); conf = float(probs[idx])
            self.votes.append(idx)
            if len(self.votes) == 5:
                best, _ = Counter(self.votes).most_common(1)[0]
                if conf >= _cfg["threshold"]:
                    out["prediction"] = PHRASE_LABELS[best]; out["confidence"] = round(conf,3)
        return out

    def close(self): self.mp.close()


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION: GESTURE  (secondary)
# ══════════════════════════════════════════════════════════════════════════════
class GestureSession:
    def __init__(self):
        self.mp   = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                                   min_detection_confidence=0.65, min_tracking_confidence=0.5)
        self.hist = deque(maxlen=10)

    def process(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); rgb.flags.writeable = False
        res = self.mp.process(rgb); rgb.flags.writeable = True
        out = {"mode":"gesture","prediction":"","confidence":0.0,"boxes":[],
               "seq_progress":10,"sentence":"","hold_count":0}

        if not res.multi_hand_landmarks:
            self.hist.clear(); return out

        hand    = res.multi_hand_landmarks[0]
        lm_list = _lm_px(bgr, hand)
        feat    = _normalize_kinivi(lm_list)
        idx, _  = _tflite_run(feat)
        self.hist.append(idx)
        h,w     = bgr.shape[:2]
        xs = [lm.x*w for lm in hand.landmark]; ys = [lm.y*h for lm in hand.landmark]
        out["boxes"] = [[max(0,int(min(xs))-28),max(0,int(min(ys))-28),
                         min(w,int(max(xs))+28),min(h,int(max(ys))+28)]]

        if len(self.hist) >= 5:
            best, cnt = Counter(self.hist).most_common(1)[0]
            sc = cnt / len(self.hist)
            if sc >= 0.5:
                out["prediction"] = KINIVI_LABELS[best] if best < len(KINIVI_LABELS) else "?"
                out["confidence"] = round(sc, 3)
        return out

    def close(self): self.mp.close()


# ══════════════════════════════════════════════════════════════════════════════
#  FastAPI
# ══════════════════════════════════════════════════════════════════════════════
app = FastAPI(title="ASL Vision Backend", version="5.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/api/status")
def status():
    return {
        "ok": True, "version": "5.1",
        "alpha_labels": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        "phrase_labels": PHRASE_LABELS,
        "gesture_labels": KINIVI_LABELS,
        "alpha_hold_frames": _cfg["alpha_hold_frames"],
        "config": _cfg,
    }


class Cfg(BaseModel):
    threshold:         float | None = None
    alpha_hold_frames: int   | None = None

@app.post("/api/settings")
def update_cfg(body: Cfg):
    if body.threshold         is not None: _cfg["threshold"]         = body.threshold
    if body.alpha_hold_frames is not None: _cfg["alpha_hold_frames"] = body.alpha_hold_frames
    return {"ok": True, "config": _cfg}


_factories = {
    "alphabet": AlphabetSession,
    "phrase":   PhraseSession,
    "gesture":  GestureSession,
}

@app.websocket("/ws/feed")
async def ws_feed(ws: WebSocket, mode: str = Query(default="alphabet")):
    await ws.accept()
    session = _factories.get(mode, AlphabetSession)()
    log.info("WS connected  mode=%s", mode)
    try:
        while True:
            raw = await ws.receive_text()
            if "," in raw: raw = raw.split(",", 1)[1]
            arr   = np.frombuffer(base64.b64decode(raw), dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None: continue

            # Handle clear command for alphabet
            if mode == "alphabet" and hasattr(session, 'clear'):
                pass  # clear is called via separate message below

            result = await asyncio.get_event_loop().run_in_executor(
                None, session.process, frame)
            await ws.send_text(json.dumps(result))

    except WebSocketDisconnect:
        log.info("WS disconnected  mode=%s", mode)
    except Exception as e:
        log.exception("WS error mode=%s: %s", mode, e)
    finally:
        session.close()
