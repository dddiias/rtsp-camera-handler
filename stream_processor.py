from __future__ import annotations

import os
import sys
import io
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from collections import deque
from typing import Optional, Tuple, Dict, Any, Deque

# =========================
# 0) Подавление логов ffmpeg/h264 ДО импорта cv2
# =========================

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000|buffer_size;1024000|loglevel;quiet",
)
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "quiet")
os.environ.setdefault("FFREPORT", "")

class H264ErrorFilter:
    """Фильтр для подавления h264/ffmpeg мусора из stderr."""
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr

    def write(self, text):
        if not text:
            return
        t = text.lower()

        # максимально агрессивно глушим h264 мусор
        if "h264" in t:
            if any(k in t for k in [
                "error", "decode", "mb", "bytestream", "cabac",
                "missing", "reference", "picture", "reorder",
                "mmco", "unref", "short", "failure", "illegal",
                "buffer", "state"
            ]):
                return

        if "ffmpeg" in t and "error" in t:
            return

        if any(k in t for k in ["decoding mb", "bytestream", "decode error"]):
            return

        self.original_stderr.write(text)

    def flush(self):
        self.original_stderr.flush()

    def __getattr__(self, name):
        return getattr(self.original_stderr, name)

if not isinstance(sys.stderr, H264ErrorFilter):
    sys.stderr = H264ErrorFilter(sys.stderr)

# =========================
# 1) Импорты после подавления логов
# =========================
import cv2
import numpy as np
import httpx


# =========================
# 2) Env loader (app.env)
# =========================

def _load_env_vars() -> Optional[str]:
    """Загружает переменные окружения из app.env с override=True."""
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(os.path.dirname(__file__), "app.env")
        if os.path.exists(env_path):
            load_dotenv(env_path, override=True)
            return env_path
        else:
            print(f"[STREAM] WARNING: app.env not found at {env_path}, using system env vars")
            return None
    except ImportError:
        print("[STREAM] WARNING: python-dotenv not installed, using system env vars only")
        return None

_env_path = _load_env_vars()
if _env_path:
    print(f"[STREAM] Loaded environment from: {_env_path}")


def _get_env_float(key: str, fallback_key: str | None = None, default: float = 0.6) -> float:
    v = os.getenv(key)
    if v is None and fallback_key:
        v = os.getenv(fallback_key)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        print(f"[STREAM] WARNING: Invalid float value for {key}: {v}, using default {default}")
        return default


def _get_env_str(key: str, fallback_key: str | None = None, default: str = "down") -> str:
    v = os.getenv(key)
    if v is None and fallback_key:
        v = os.getenv(fallback_key)
    if v is None:
        return default
    return str(v).strip().lower()


# =========================
# 3) Настройки
# =========================

PLATE_CAMERA_RTSP = os.getenv("PLATE_CAMERA_RTSP", "rtsp://USER:PASSWORD@HOST:554/streaming/channels/101")
SNOW_CAMERA_RTSP  = os.getenv("SNOW_CAMERA_RTSP",  "rtsp://USER:PASSWORD@HOST:555/streaming/channels/101")

PLATE_LINE_Y_POSITION = _get_env_float("PLATE_LINE_Y_POSITION", "LINE_Y_POSITION", 0.6)
PLATE_LINE_DIRECTION  = _get_env_str("PLATE_LINE_DIRECTION", "LINE_DIRECTION", "down")

SNOW_LINE_Y_POSITION  = _get_env_float("SNOW_LINE_Y_POSITION", "LINE_Y_POSITION", 0.6)
SNOW_LINE_DIRECTION   = _get_env_str("SNOW_LINE_DIRECTION", "LINE_DIRECTION", "down")

print(f"[STREAM] Plate camera line: Y={PLATE_LINE_Y_POSITION}, direction={PLATE_LINE_DIRECTION}")
print(f"[STREAM] Snow camera line:  Y={SNOW_LINE_Y_POSITION}, direction={SNOW_LINE_DIRECTION}")

MIN_CONFIDENCE = float(os.getenv("STREAM_MIN_CONFIDENCE", "0.5"))
MIN_BBOX_AREA  = int(os.getenv("STREAM_MIN_BBOX_AREA", "10000"))

TRACK_MAX_AGE          = int(os.getenv("TRACK_MAX_AGE", "30"))   # возраст трека в "тикающих" апдейтах
TRACK_MIN_HITS         = int(os.getenv("TRACK_MIN_HITS", "3"))
TRACK_IOU_THRESHOLD    = float(os.getenv("TRACK_IOU_THRESHOLD", "0.3"))
TRACK_CROSS_COOLDOWN_S = float(os.getenv("TRACK_CROSS_COOLDOWN_S", "1.0"))

DEDUP_WINDOW_SECONDS = float(os.getenv("STREAM_DEDUP_WINDOW_SECONDS", "5.0"))

SHOW_STREAM_WINDOW = os.getenv("SHOW_STREAM_WINDOW", "false").strip().lower() == "true"
DETECTION_INTERVAL = int(os.getenv("STREAM_DETECTION_INTERVAL", "3"))

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")

UPSTREAM_URL = os.getenv("UPSTREAM_URL", "https://snowops-anpr-service.onrender.com/api/v1/anpr/events")
PLATE_CAMERA_ID = os.getenv("PLATE_CAMERA_ID", "camera-001")


def _silence_opencv_logs() -> None:
    """Глушим логи OpenCV/FFmpeg; учитываем разные версии OpenCV."""
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
        return
    except Exception:
        pass
    try:
        cv2.setLogLevel(cv2.LOG_LEVEL_SILENT)
    except Exception:
        pass

_silence_opencv_logs()


# =========================
# 4) Модели данных
# =========================

@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]      # x1,y1,x2,y2
    center: Tuple[int, int]              # cx,cy
    confidence: float
    age: int                             # сколько апдейтов не видели (или общий возраст)
    hits: int                            # сколько раз матчился
    last_seen_ts: float                  # time.time()
    crossed: bool
    direction: Optional[str]
    last_cross_ts: float = 0.0           # для cooldown


@dataclass
class TimestampedFrame:
    frame: np.ndarray
    timestamp: float


# =========================
# 5) Логика пересечения линии
# =========================

class LineCrossingDetector:
    """
    Простенький IOU-трекер + детектор пересечения горизонтальной линии
    по движению центра bbox.
    """
    def __init__(self, line_y_ratio: float, direction: str = "down"):
        self.line_y_ratio = float(line_y_ratio)
        self.direction = direction.strip().lower()
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1

        self.frame_height = 0
        self.frame_width = 0

    @staticmethod
    def _iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        inter = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def process_frame(self, frame: np.ndarray, detections: list[tuple[int, int, int, int, float]]) -> list[Track]:
        self.frame_height, self.frame_width = frame.shape[:2]
        now_ts = time.time()
        if self.frame_height == 0 or self.frame_width == 0:
            return []

        line_y = int(self.frame_height * self.line_y_ratio)

        # состариваем треки
        for tr in self.tracks.values():
            tr.age += 1

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        # матчим сначала более "надёжные" треки
        sorted_tracks = sorted(self.tracks.items(), key=lambda kv: (kv[1].hits, -kv[1].age), reverse=True)

        crossed_now_tracks: list[Track] = []

        for track_id, tr in sorted_tracks:
            best_iou = 0.0
            best_det_idx: Optional[int] = None

            for det_idx, (x1, y1, x2, y2, conf) in enumerate(detections):
                if det_idx in matched_dets:
                    continue
                iou = self._iou(tr.bbox, (x1, y1, x2, y2))
                if iou > best_iou and iou >= TRACK_IOU_THRESHOLD:
                    best_iou = iou
                    best_det_idx = det_idx

            if best_det_idx is None:
                continue

            x1, y1, x2, y2, conf = detections[best_det_idx]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            prev_cy = tr.center[1]
            crossed_now = False

            # детектим пересечение только если ещё не crossed и соблюдён cooldown
            if (not tr.crossed) and (now_ts - tr.last_cross_ts >= TRACK_CROSS_COOLDOWN_S):
                if self.direction == "down":
                    if prev_cy < line_y <= cy:
                        crossed_now = True
                else:  # up
                    if prev_cy > line_y >= cy:
                        crossed_now = True

            tr.bbox = (x1, y1, x2, y2)
            tr.center = (cx, cy)
            tr.confidence = conf
            tr.age = 0
            tr.hits += 1
            tr.last_seen_ts = now_ts

            if crossed_now:
                tr.crossed = True
                tr.direction = self.direction
                tr.last_cross_ts = now_ts
                crossed_now_tracks.append(tr)

            matched_tracks.add(track_id)
            matched_dets.add(best_det_idx)

        # новые треки
        for det_idx, (x1, y1, x2, y2, conf) in enumerate(detections):
            if det_idx in matched_dets:
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            tr = Track(
                track_id=self.next_track_id,
                bbox=(x1, y1, x2, y2),
                center=(cx, cy),
                confidence=conf,
                age=0,
                hits=1,
                last_seen_ts=now_ts,
                crossed=False,
                direction=None,
                last_cross_ts=0.0,
            )
            self.tracks[self.next_track_id] = tr
            self.next_track_id += 1

        # удаляем старые
        to_remove = []
        for track_id, tr in self.tracks.items():
            if tr.age > TRACK_MAX_AGE:
                to_remove.append(track_id)
            elif tr.hits < TRACK_MIN_HITS and tr.age > 5:
                to_remove.append(track_id)
        for tid in to_remove:
            self.tracks.pop(tid, None)

        return crossed_now_tracks


# =========================
# 6) StreamProcessor
# =========================

class StreamProcessor:
    """
    Два потока:
      - snow thread: держит буфер кадров и буфер кадров "пересечения" (центр пересёк линию)
      - plate thread: ловит пересечение по номерной камере и кладёт задачу в worker
      - worker thread: делает Gemini + POST upstream (не блокируя видеопоток)
    """
    def __init__(self, merger):
        self.merger = merger

        # reload env
        _load_env_vars()

        plate_y = _get_env_float("PLATE_LINE_Y_POSITION", "LINE_Y_POSITION", 0.6)
        plate_dir = _get_env_str("PLATE_LINE_DIRECTION", "LINE_DIRECTION", "down")

        snow_y = _get_env_float("SNOW_LINE_Y_POSITION", "LINE_Y_POSITION", 0.6)
        snow_dir = _get_env_str("SNOW_LINE_DIRECTION", "LINE_DIRECTION", "down")

        self.plate_detector = LineCrossingDetector(plate_y, plate_dir)
        self.snow_detector  = LineCrossingDetector(snow_y, snow_dir)

        print("[STREAM] Initialized detectors:")
        print(f"[STREAM]   Plate: Y={plate_y}, direction={plate_dir}")
        print(f"[STREAM]   Snow:  Y={snow_y}, direction={snow_dir}")

        self.plate_cap: Optional[cv2.VideoCapture] = None
        self.snow_cap: Optional[cv2.VideoCapture] = None

        self._stop_event = threading.Event()

        self._snow_thread: Optional[threading.Thread] = None
        self._plate_thread: Optional[threading.Thread] = None
        self._worker_thread: Optional[threading.Thread] = None

        # snow frame buffers
        self._snow_frame_buffer: Deque[TimestampedFrame] = deque(maxlen=120)
        self._snow_crossing_frames: Deque[dict] = deque(maxlen=20)
        self._snow_buffer_lock = threading.Lock()
        self._snow_cross_lock = threading.Lock()

        # dedup by plate from gemini
        self._processed_plates: Dict[str, float] = {}
        self._plates_lock = threading.Lock()

        # tasks to worker
        self._task_queue: "deque[dict]" = deque()
        self._task_lock = threading.Lock()
        self._task_signal = threading.Event()

        # yolo
        self.yolo_model = None
        self._yolo_lock = threading.Lock()
        self._load_yolo_model()

    def _load_yolo_model(self) -> None:
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(YOLO_MODEL_PATH)
            print(f"[STREAM] YOLO model loaded: {YOLO_MODEL_PATH}")
        except Exception as e:
            print(f"[STREAM] ERROR: Failed to load YOLO model: {e}")
            self.yolo_model = None

    def _detect_vehicles(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        if self.yolo_model is None:
            return []
        try:
            # COCO: car=2, truck=7
            with self._yolo_lock:
                results = self.yolo_model(frame, classes=[2, 7], conf=MIN_CONFIDENCE, verbose=False)

            detections = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0].item())
                    area = (x2 - x1) * (y2 - y1)
                    if area < MIN_BBOX_AREA:
                        continue
                    detections.append((x1, y1, x2, y2, conf))
            return detections
        except Exception as e:
            print(f"[STREAM] Error in vehicle detection: {e}")
            return []

    # ------------- snow frame selection -------------

    def _validate_frame(self, frame: np.ndarray) -> bool:
        try:
            if frame is None or frame.size == 0:
                return False
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                return False
            if frame.dtype != np.uint8:
                return False
            if np.all(frame == 0):
                return False
            if np.any(np.isnan(frame)):
                return False
            return True
        except Exception:
            return False

    def _get_snow_frame(self, prefer_crossing: bool = True) -> Optional[np.ndarray]:
        now_ts = time.time()

        if prefer_crossing:
            with self._snow_cross_lock:
                # чистим старьё
                while self._snow_crossing_frames and (now_ts - self._snow_crossing_frames[0]["timestamp"] > 2.0):
                    self._snow_crossing_frames.popleft()

                if self._snow_crossing_frames:
                    item = self._snow_crossing_frames[-1]
                    fr = item["frame"]
                    if self._validate_frame(fr):
                        print(f"[STREAM] Using snow frame from crossing buffer (track_id={item.get('track_id')})")
                        return fr.copy()

        with self._snow_buffer_lock:
            if not self._snow_frame_buffer:
                return None
            for i in range(len(self._snow_frame_buffer) - 1, -1, -1):
                fr = self._snow_frame_buffer[i].frame
                if self._validate_frame(fr):
                    return fr.copy()

        return None

    def _encode_frame_to_jpeg(self, frame: np.ndarray) -> Optional[bytes]:
        if not self._validate_frame(frame):
            return None
        try:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                return None
            b = buf.tobytes()
            return b if b else None
        except Exception as e:
            print(f"[STREAM] Error encoding frame: {e}")
            return None

    # ------------- drawing -------------

    def _draw_line_and_tracks(self, frame: np.ndarray, detector: LineCrossingDetector, detections: list, title: str) -> np.ndarray:
        if not SHOW_STREAM_WINDOW:
            return frame

        out = frame.copy()
        h, w = out.shape[:2]
        line_y = int(h * detector.line_y_ratio)
        color = (0, 255, 0) if detector.direction == "down" else (0, 0, 255)

        cv2.line(out, (0, line_y), (w, line_y), color, 2)
        cv2.putText(out, f"Line ({detector.direction})", (10, max(20, line_y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        for x1, y1, x2, y2, conf in detections:
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(out, f"{conf:.2f}", (x1, max(15, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        for tr in detector.tracks.values():
            x1, y1, x2, y2 = tr.bbox
            cx, cy = tr.center
            tcolor = (0, 255, 0) if tr.crossed else (0, 165, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), tcolor, 2)
            cv2.circle(out, (cx, cy), 5, tcolor, -1)
            cv2.putText(out, f"ID:{tr.track_id}", (x1, max(15, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, tcolor, 1)
            if tr.crossed:
                cv2.putText(out, "CROSSED", (x1, min(h - 10, y2 + 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, tcolor, 2)

        cv2.putText(out, title, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return out

    # ------------- worker queue -------------

    def _push_task(self, task: dict) -> None:
        with self._task_lock:
            self._task_queue.append(task)
            self._task_signal.set()

    def _pop_task(self) -> Optional[dict]:
        with self._task_lock:
            if not self._task_queue:
                self._task_signal.clear()
                return None
            return self._task_queue.popleft()

    # ------------- processing -------------

    async def _process_crossing_async(self, plate_frame: np.ndarray) -> None:
        """
        Делает:
          - берём snow_frame (предпочтительно из crossing buffer)
          - JPEG encode
          - Gemini analyze (plate + snow)
          - dedup по plate от Gemini
          - POST upstream multipart (event + 2 фото)
        """
        now_ts = time.time()

        snow_frame = self._get_snow_frame(prefer_crossing=True)
        if snow_frame is None:
            print("[STREAM] No snow frame available, skipping")
            return

        plate_bytes = self._encode_frame_to_jpeg(plate_frame)
        snow_bytes  = self._encode_frame_to_jpeg(snow_frame)
        if plate_bytes is None or snow_bytes is None:
            print("[STREAM] Failed to encode frames, skipping")
            return

        # Gemini
        gemini_result: dict | None = None
        try:
            gemini_result = await self.merger.analyze_with_gemini(
                snow_photo=snow_bytes,
                plate_photo_1=plate_bytes,
                plate_photo_2=None,
                camera_plate=None,
            )
            print(f"[STREAM] Gemini result: {gemini_result}")
        except Exception as e:
            print(f"[STREAM] Gemini error: {e}")
            return

        plate = (gemini_result or {}).get("plate")
        plate_conf = float((gemini_result or {}).get("plate_confidence", 0.0) or 0.0)

        # dedup
        if plate:
            plate = str(plate).strip().upper()
            with self._plates_lock:
                # clean old
                old = [p for p, ts in self._processed_plates.items() if (now_ts - ts) > DEDUP_WINDOW_SECONDS]
                for p in old:
                    self._processed_plates.pop(p, None)

                if plate in self._processed_plates:
                    print(f"[STREAM] Duplicate plate (Gemini): {plate}, skipping")
                    return

                self._processed_plates[plate] = now_ts

        now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        event_data = {
            "camera_id": PLATE_CAMERA_ID,
            "event_time": now_iso,
            "plate": plate,
            "confidence": plate_conf,
            "direction": self.plate_detector.direction,
            "lane": 0,
            "vehicle": {},
            "plate_source": "gemini",
            "snow_volume_percentage": float((gemini_result or {}).get("snow_percentage", 0.0) or 0.0),
            "snow_volume_confidence": float((gemini_result or {}).get("snow_confidence", 0.0) or 0.0),
            "matched_snow": True,
            "gemini_result": gemini_result,
            "timestamp": now_iso,
        }

        # upstream
        try:
            data = {"event": json.dumps(event_data, ensure_ascii=False)}
            files = [
                ("photos", ("detectionPicture.jpg", plate_bytes, "image/jpeg")),
                ("photos", ("snowSnapshot.jpg", snow_bytes, "image/jpeg")),
            ]
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(UPSTREAM_URL, data=data, files=files)
            print(f"[STREAM] Upstream: status={resp.status_code}, ok={resp.is_success}, body={resp.text[:200]}")
        except Exception as e:
            print(f"[STREAM] Upstream send error: {e}")

    def _worker_loop(self):
        """
        Worker поток:
          - ждёт tasks
          - запускает asyncio.run() для async обработки (последовательно)
        """
        print("[STREAM] Worker thread started")
        while not self._stop_event.is_set():
            if not self._task_signal.wait(timeout=0.2):
                continue

            task = self._pop_task()
            if task is None:
                continue

            plate_frame = task.get("plate_frame")
            if plate_frame is None:
                continue

            try:
                import asyncio
                asyncio.run(self._process_crossing_async(plate_frame))
            except Exception as e:
                print(f"[STREAM] Worker error: {e}")

        print("[STREAM] Worker thread stopped")

    def _snow_processing_loop(self):
        print("[STREAM] Starting snow stream processing...")
        self.snow_cap = cv2.VideoCapture(SNOW_CAMERA_RTSP, cv2.CAP_FFMPEG)
        if not self.snow_cap.isOpened():
            print(f"[STREAM] ERROR: Cannot open snow camera: {SNOW_CAMERA_RTSP}")
            return

        try:
            self.snow_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        fail_count = 0
        MAX_FAILS = 15

        window_created = False
        if SHOW_STREAM_WINDOW:
            try:
                cv2.namedWindow("Snow Camera - Line Crossing", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Snow Camera - Line Crossing", 1280, 720)
                window_created = True
                print("[STREAM] Snow display window created")
            except Exception as e:
                print(f"[STREAM] Snow display window failed: {e}")

        frame_counter = 0

        while not self._stop_event.is_set():
            ret, frame = False, None
            try:
                ret, frame = self.snow_cap.read()
            except Exception:
                ret, frame = False, None

            if not ret or frame is None or frame.size == 0 or len(frame.shape) != 3:
                fail_count += 1
                if fail_count >= MAX_FAILS:
                    print("[STREAM] Snow camera reconnecting...")
                    try:
                        self.snow_cap.release()
                    except Exception:
                        pass
                    time.sleep(2)
                    self.snow_cap = cv2.VideoCapture(SNOW_CAMERA_RTSP, cv2.CAP_FFMPEG)
                    fail_count = 0
                time.sleep(0.05)
                continue

            fail_count = 0
            now_ts = time.time()

            # общий буфер
            with self._snow_buffer_lock:
                while self._snow_frame_buffer and (now_ts - self._snow_frame_buffer[0].timestamp > 3.0):
                    self._snow_frame_buffer.popleft()
                self._snow_frame_buffer.append(TimestampedFrame(frame=frame.copy(), timestamp=now_ts))

            frame_counter += 1
            detections = []
            crossed = []

            if frame_counter % max(1, DETECTION_INTERVAL) == 0:
                detections = self._detect_vehicles(frame)
                if detections:
                    crossed = self.snow_detector.process_frame(frame, detections)

                # если пересёк — кладём кадр в crossing buffer
                for tr in crossed:
                    h, _w = frame.shape[:2]
                    line_y = int(h * self.snow_detector.line_y_ratio)
                    cy = tr.center[1]
                    ok = (self.snow_detector.direction == "down" and cy >= line_y) or (
                        self.snow_detector.direction == "up" and cy <= line_y
                    )
                    if ok:
                        with self._snow_cross_lock:
                            self._snow_crossing_frames.append({
                                "frame": frame.copy(),
                                "timestamp": now_ts,
                                "track_id": tr.track_id,
                            })
                        print(f"[STREAM] Snow center crossing: track_id={tr.track_id}, cy={cy}, line_y={line_y}")

            if SHOW_STREAM_WINDOW and window_created:
                try:
                    show = self._draw_line_and_tracks(frame, self.snow_detector, detections, "SNOW")
                    cv2.imshow("Snow Camera - Line Crossing", show)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        self._stop_event.set()
                        break
                except Exception:
                    window_created = False

            time.sleep(0.01)

        if SHOW_STREAM_WINDOW and window_created:
            try:
                cv2.destroyWindow("Snow Camera - Line Crossing")
            except Exception:
                pass

        try:
            self.snow_cap.release()
        except Exception:
            pass

        print("[STREAM] Snow processing loop stopped")

    def _plate_processing_loop(self):
        print("[STREAM] Starting plate stream processing...")
        self.plate_cap = cv2.VideoCapture(PLATE_CAMERA_RTSP, cv2.CAP_FFMPEG)
        if not self.plate_cap.isOpened():
            print(f"[STREAM] ERROR: Cannot open plate camera: {PLATE_CAMERA_RTSP}")
            return

        try:
            self.plate_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        fail_count = 0
        MAX_FAILS = 15

        window_created = False
        if SHOW_STREAM_WINDOW:
            try:
                cv2.namedWindow("Plate Camera - Line Crossing", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Plate Camera - Line Crossing", 1280, 720)
                window_created = True
                print("[STREAM] Plate display window created")
            except Exception as e:
                print(f"[STREAM] Plate display window failed: {e}")

        frame_counter = 0

        while not self._stop_event.is_set():
            ret, frame = False, None
            try:
                ret, frame = self.plate_cap.read()
            except Exception:
                ret, frame = False, None

            if not ret or frame is None or frame.size == 0 or len(frame.shape) != 3:
                fail_count += 1
                if fail_count >= MAX_FAILS:
                    print("[STREAM] Plate camera reconnecting...")
                    try:
                        self.plate_cap.release()
                    except Exception:
                        pass
                    time.sleep(2)
                    self.plate_cap = cv2.VideoCapture(PLATE_CAMERA_RTSP, cv2.CAP_FFMPEG)
                    fail_count = 0
                time.sleep(0.05)
                continue

            fail_count = 0

            frame_counter += 1
            detections = []
            crossed = []

            if frame_counter % max(1, DETECTION_INTERVAL) == 0:
                detections = self._detect_vehicles(frame)
                if detections:
                    crossed = self.plate_detector.process_frame(frame, detections)

            # если пересёк — кладём задачу в worker (НЕ блокируем видеопоток)
            for tr in crossed:
                print(f"[STREAM] Plate crossing: track_id={tr.track_id}, bbox={tr.bbox}")
                self._push_task({"plate_frame": frame.copy()})

            if SHOW_STREAM_WINDOW and window_created:
                try:
                    show = self._draw_line_and_tracks(frame, self.plate_detector, detections, "PLATE")
                    cv2.imshow("Plate Camera - Line Crossing", show)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        self._stop_event.set()
                        break
                except Exception:
                    window_created = False

            time.sleep(0.01)

        if SHOW_STREAM_WINDOW and window_created:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        try:
            self.plate_cap.release()
        except Exception:
            pass

        print("[STREAM] Plate processing loop stopped")

    # =========================
    # Public API
    # =========================

    def start(self) -> None:
        if self._plate_thread and self._plate_thread.is_alive():
            print("[STREAM] Already running")
            return

        self._stop_event.clear()

        # worker
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="stream-worker")
        self._worker_thread.start()

        # snow
        self._snow_thread = threading.Thread(target=self._snow_processing_loop, daemon=True, name="snow-processor")
        self._snow_thread.start()

        # дать снегу набрать буфер
        time.sleep(2)

        # plate
        self._plate_thread = threading.Thread(target=self._plate_processing_loop, daemon=True, name="plate-processor")
        self._plate_thread.start()

        print("[STREAM] Stream processor started (snow + plate + worker)")

    def stop(self) -> None:
        self._stop_event.set()

        # разбудим worker
        self._task_signal.set()

        for th in [self._plate_thread, self._snow_thread, self._worker_thread]:
            if th:
                th.join(timeout=5)

        self._plate_thread = None
        self._snow_thread = None
        self._worker_thread = None
        print("[STREAM] Stream processor stopped")


# =========================
# Singleton helpers
# =========================

_stream_processor: Optional[StreamProcessor] = None

def init_stream_processor(merger) -> StreamProcessor:
    global _stream_processor
    if _stream_processor is None:
        _stream_processor = StreamProcessor(merger)
    return _stream_processor

def get_stream_processor() -> Optional[StreamProcessor]:
    return _stream_processor
