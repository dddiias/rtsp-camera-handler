import os
import sys
import io
import threading
import time
from typing import Optional, Tuple, Dict, Any
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import asyncio
import json

# Подавляем логи h264 ДО импорта cv2
# Принудительно настраиваем FFMPEG backend с максимальным подавлением логов
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000|buffer_size;1024000|loglevel;quiet|err_detect;ignore_err",
)

# Глушим логи FFmpeg/h264 через переменные окружения (до импорта cv2)
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "quiet"
os.environ["OPENCV_FFMPEG_DLL_DIR"] = ""  # Убираем лишние логи
# Подавляем все логи FFmpeg
os.environ["FFREPORT"] = ""  # Отключаем отчеты FFmpeg

# Перенаправляем stderr ДО импорта cv2 для подавления h264 ошибок
class H264ErrorFilter:
    """Фильтр для подавления h264 ошибок из stderr"""
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
    
    def write(self, text):
        # Фильтруем все сообщения, содержащие h264 ошибки
        text_lower = text.lower()
        if "h264" in text_lower and ("error" in text_lower or "decode" in text_lower or "mb" in text_lower or "bytestream" in text_lower):
            return  # Полностью игнорируем h264 ошибки
        # Также фильтруем другие FFmpeg ошибки
        if "ffmpeg" in text_lower and "error" in text_lower:
            return
        # Пропускаем остальные сообщения
        self.original_stderr.write(text)
    
    def flush(self):
        self.original_stderr.flush()
    
    def __getattr__(self, name):
        return getattr(self.original_stderr, name)

# Применяем фильтр ДО импорта cv2
if not isinstance(sys.stderr, H264ErrorFilter):
    sys.stderr = H264ErrorFilter(sys.stderr)

# Теперь импортируем cv2 и другие модули
import cv2
import numpy as np
import httpx

# Загружаем переменные окружения из app.env
def _load_env_vars():
    """Загружает переменные окружения из app.env с перезаписью"""
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(os.path.dirname(__file__), "app.env")
        if os.path.exists(env_path):
            # Используем override=True чтобы перезагрузить переменные
            load_dotenv(env_path, override=True)
            return env_path
        else:
            print(f"[STREAM] WARNING: app.env not found at {env_path}, using system env vars")
            return None
    except ImportError:
        print(f"[STREAM] WARNING: python-dotenv not installed, using system env vars only")
        return None

# Загружаем переменные окружения при старте
_env_path = _load_env_vars()
if _env_path:
    print(f"[STREAM] Loaded environment from: {_env_path}")

# Функция для получения значений с перезагрузкой
def _get_env_float(key: str, fallback_key: str = None, default: float = 0.6) -> float:
    """Получает float значение из переменных окружения"""
    value = os.getenv(key)
    if value is None and fallback_key:
        value = os.getenv(fallback_key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        print(f"[STREAM] WARNING: Invalid float value for {key}: {value}, using default {default}")
        return default

def _get_env_str(key: str, fallback_key: str = None, default: str = "down") -> str:
    """Получает string значение из переменных окружения"""
    value = os.getenv(key)
    if value is None and fallback_key:
        value = os.getenv(fallback_key)
    if value is None:
        return default
    return value.lower()

# Настройки через переменные окружения
PLATE_CAMERA_RTSP = os.getenv(
    "PLATE_CAMERA_RTSP",
    "rtsp://admin:Armat456321@178.22.170.254:554/streaming/channels/101"
)
SNOW_CAMERA_RTSP = os.getenv(
    "SNOW_CAMERA_RTSP",
    "rtsp://admin:Armat456321@178.22.170.254:555/streaming/channels/101"
)

# Настройки детекции пересечения линии для номерной камеры
PLATE_LINE_Y_POSITION = _get_env_float("PLATE_LINE_Y_POSITION", "LINE_Y_POSITION", 0.6)
PLATE_LINE_DIRECTION = _get_env_str("PLATE_LINE_DIRECTION", "LINE_DIRECTION", "down")

# Настройки детекции пересечения линии для снеговой камеры
SNOW_LINE_Y_POSITION = _get_env_float("SNOW_LINE_Y_POSITION", "LINE_Y_POSITION", 0.6)
SNOW_LINE_DIRECTION = _get_env_str("SNOW_LINE_DIRECTION", "LINE_DIRECTION", "down")

# Выводим текущие значения для отладки
print(f"[STREAM] Plate camera line: Y={PLATE_LINE_Y_POSITION}, direction={PLATE_LINE_DIRECTION}")
print(f"[STREAM] Snow camera line: Y={SNOW_LINE_Y_POSITION}, direction={SNOW_LINE_DIRECTION}")

# Общие настройки детекции
MIN_CONFIDENCE = float(os.getenv("STREAM_MIN_CONFIDENCE", "0.5"))
MIN_BBOX_AREA = int(os.getenv("STREAM_MIN_BBOX_AREA", "10000"))

# Настройки трекинга
TRACK_MAX_AGE = int(os.getenv("TRACK_MAX_AGE", "30"))  # Максимальный возраст трека в кадрах
TRACK_MIN_HITS = int(os.getenv("TRACK_MIN_HITS", "3"))  # Минимальное количество попаданий для создания трека
TRACK_IOU_THRESHOLD = float(os.getenv("TRACK_IOU_THRESHOLD", "0.3"))

# Настройки дедупликации
DEDUP_WINDOW_SECONDS = float(os.getenv("STREAM_DEDUP_WINDOW_SECONDS", "5.0"))

# Отображение потоков (для отладки)
SHOW_STREAM_WINDOW = os.getenv("SHOW_STREAM_WINDOW", "false").lower() == "true"

def _silence_opencv_logs():
    """Глушим логи OpenCV/FFmpeg и h264"""
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        try:
            cv2.setLogLevel(cv2.LOG_LEVEL_SILENT)
        except Exception:
            pass

_silence_opencv_logs()


@dataclass
class Track:
    """Трек для отслеживания объекта"""
    track_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]  # center_x, center_y
    confidence: float
    age: int
    hits: int
    last_seen: float  # timestamp
    crossed: bool  # Пересекла ли линию
    direction: Optional[str]  # "down" или "up"


@dataclass
class TimestampedFrame:
    """Кадр с временной меткой"""
    frame: np.ndarray
    timestamp: float


class LineCrossingDetector:
    """Детектор пересечения линии"""
    
    def __init__(self, line_y_ratio: float, direction: str = "down"):
        """
        Args:
            line_y_ratio: Позиция линии (0.0-1.0, где 0.0 = верх, 1.0 = низ)
            direction: Направление пересечения ("down" = сверху вниз, "up" = снизу вверх)
        """
        self.line_y_ratio = line_y_ratio
        self.direction = direction
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.frame_height = 0
        self.frame_width = 0
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Вычисляет IoU (Intersection over Union) между двумя боксами"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Вычисляем площадь пересечения
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Вычисляем площадь объединения
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _update_tracks(self, detections: list) -> list:
        """
        Обновляет треки на основе новых детекций
        
        Args:
            detections: Список детекций [(x1, y1, x2, y2, conf), ...]
        
        Returns:
            Список обновленных треков
        """
        if self.frame_height == 0 or self.frame_width == 0:
            return []
        
        line_y_px = int(self.frame_height * self.line_y_ratio)
        
        # Обновляем возраст всех треков
        for track in self.tracks.values():
            track.age += 1
        
        # Сопоставляем детекции с существующими треками
        matched_tracks = set()
        matched_detections = set()
        
        # Сортируем треки по количеству попаданий (приоритет более надежным)
        sorted_tracks = sorted(
            self.tracks.items(),
            key=lambda x: (x[1].hits, -x[1].age),
            reverse=True
        )
        
        for track_id, track in sorted_tracks:
            if track_id in matched_tracks:
                continue
            
            best_iou = 0.0
            best_detection_idx = None
            
            for det_idx, (x1, y1, x2, y2, conf) in enumerate(detections):
                if det_idx in matched_detections:
                    continue
                
                det_bbox = (x1, y1, x2, y2)
                iou = self._calculate_iou(track.bbox, det_bbox)
                
                if iou > best_iou and iou > TRACK_IOU_THRESHOLD:
                    best_iou = iou
                    best_detection_idx = det_idx
            
            if best_detection_idx is not None:
                # Обновляем трек
                x1, y1, x2, y2, conf = detections[best_detection_idx]
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Проверяем пересечение линии
                prev_center_y = track.center[1]
                crossed = False
                
                if not track.crossed:
                    if self.direction == "down":
                        # Пересечение сверху вниз: центр был выше линии, теперь ниже или на линии
                        if prev_center_y < line_y_px and center_y >= line_y_px:
                            crossed = True
                    else:  # up
                        # Пересечение снизу вверх: центр был ниже линии, теперь выше или на линии
                        if prev_center_y > line_y_px and center_y <= line_y_px:
                            crossed = True
                
                track.bbox = (x1, y1, x2, y2)
                track.center = (center_x, center_y)
                track.confidence = conf
                track.age = 0
                track.hits += 1
                track.last_seen = time.time()
                track.crossed = track.crossed or crossed
                if crossed:
                    track.direction = self.direction
                
                matched_tracks.add(track_id)
                matched_detections.add(best_detection_idx)
        
        # Создаем новые треки для несоответствующих детекций
        for det_idx, (x1, y1, x2, y2, conf) in enumerate(detections):
            if det_idx in matched_detections:
                continue
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            new_track = Track(
                track_id=self.next_track_id,
                bbox=(x1, y1, x2, y2),
                center=(center_x, center_y),
                confidence=conf,
                age=0,
                hits=1,
                last_seen=time.time(),
                crossed=False,
                direction=None,
            )
            self.tracks[self.next_track_id] = new_track
            self.next_track_id += 1
        
        # Удаляем старые треки
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track.age > TRACK_MAX_AGE or (track.hits < TRACK_MIN_HITS and track.age > 5):
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        # Возвращаем треки, которые только что пересекли линию
        crossing_tracks = []
        for track in self.tracks.values():
            if track.crossed and track.direction == self.direction:
                # Проверяем, не пересекали ли мы уже эту линию недавно
                if track.last_seen == time.time() or (time.time() - track.last_seen) < 0.5:
                    crossing_tracks.append(track)
        
        return crossing_tracks
    
    def process_frame(self, frame: np.ndarray, detections: list) -> list:
        """
        Обрабатывает кадр и возвращает треки, которые пересекли линию
        
        Args:
            frame: Кадр изображения
            detections: Список детекций [(x1, y1, x2, y2, conf), ...]
        
        Returns:
            Список треков, которые только что пересекли линию
        """
        self.frame_height, self.frame_width = frame.shape[:2]
        return self._update_tracks(detections)


class StreamProcessor:
    """Обработчик RTSP потоков с детекцией пересечения линии"""
    
    def __init__(self, merger):
        """
        Args:
            merger: Экземпляр EventMerger для отправки в Gemini
        """
        self.merger = merger
        self.plate_cap: Optional[cv2.VideoCapture] = None
        self.snow_cap: Optional[cv2.VideoCapture] = None
        
        # Перезагружаем переменные окружения перед созданием детекторов
        _load_env_vars()
        plate_y = _get_env_float("PLATE_LINE_Y_POSITION", "LINE_Y_POSITION", 0.6)
        plate_dir = _get_env_str("PLATE_LINE_DIRECTION", "LINE_DIRECTION", "down")
        snow_y = _get_env_float("SNOW_LINE_Y_POSITION", "LINE_Y_POSITION", 0.6)
        snow_dir = _get_env_str("SNOW_LINE_DIRECTION", "LINE_DIRECTION", "down")
        
        # Детекторы для каждой камеры со своими настройками
        self.plate_detector = LineCrossingDetector(plate_y, plate_dir)
        self.snow_detector = LineCrossingDetector(snow_y, snow_dir)
        
        print(f"[STREAM] Initialized detectors:")
        print(f"[STREAM]   Plate: Y={plate_y}, direction={plate_dir}")
        print(f"[STREAM]   Snow: Y={snow_y}, direction={snow_dir}")
        self.yolo_model = None
        self._stop_event = threading.Event()
        self._processing_thread: Optional[threading.Thread] = None
        self._snow_frame_buffer: deque = deque(maxlen=90)  # ~3 секунды при 30 FPS
        self._snow_buffer_lock = threading.Lock()
        self._processed_plates: Dict[str, float] = {}  # plate -> timestamp для дедупликации (по номеру от Gemini)
        self._plates_lock = threading.Lock()
        
        # Загружаем YOLO модель
        self._load_yolo_model()
    
    def _load_yolo_model(self):
        """Загружает YOLO модель для детекции машин"""
        try:
            from ultralytics import YOLO
            model_path = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
            self.yolo_model = YOLO(model_path)
            print(f"[STREAM] YOLO model loaded: {model_path}")
        except Exception as e:
            print(f"[STREAM] ERROR: Failed to load YOLO model: {e}")
            self.yolo_model = None
    
    def _detect_vehicles(self, frame: np.ndarray) -> list:
        """Детектирует машины в кадре"""
        if self.yolo_model is None:
            return []
        
        try:
            # Класс 2 = car, класс 7 = truck в COCO dataset
            results = self.yolo_model(frame, classes=[2, 7], conf=MIN_CONFIDENCE, verbose=False)
            detections = []
            
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0].item())
                    
                    # Фильтр по минимальной площади
                    area = (x2 - x1) * (y2 - y1)
                    if area < MIN_BBOX_AREA:
                        continue
                    
                    detections.append((x1, y1, x2, y2, conf))
            
            return detections
        except Exception as e:
            print(f"[STREAM] Error in vehicle detection: {e}")
            return []
    
    def _draw_snow_line_and_tracks(self, frame: np.ndarray, detections: list, crossing_tracks: list) -> np.ndarray:
        """Отрисовывает линию пересечения, детекции и треки на снеговом кадре"""
        if not SHOW_STREAM_WINDOW:
            return frame
        
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # Рисуем линию пересечения для снеговой камеры (используем значения из детектора)
        line_y = int(h * self.snow_detector.line_y_ratio)
        color = (0, 255, 0) if self.snow_detector.direction == "down" else (0, 0, 255)
        cv2.line(display_frame, (0, line_y), (w, line_y), color, 2)
        cv2.putText(display_frame, f"Line ({self.snow_detector.direction})", (10, line_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Рисуем детекции
        for x1, y1, x2, y2, conf in detections:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(display_frame, f"{conf:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Рисуем треки и пересечения
        for track in self.snow_detector.tracks.values():
            x1, y1, x2, y2 = track.bbox
            center_x, center_y = track.center
            
            # Цвет трека: зеленый если пересек, красный если нет
            track_color = (0, 255, 0) if track.crossed else (0, 165, 255)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), track_color, 2)
            cv2.circle(display_frame, (center_x, center_y), 5, track_color, -1)
            cv2.putText(display_frame, f"ID:{track.track_id}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, track_color, 1)
            
            if track.crossed:
                cv2.putText(display_frame, "CENTER CROSSED!", (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return display_frame
    
    def _snow_processing_loop(self):
        """Обработка снегового потока с детекцией пересечения линии (когда центр грузовика пересекает)"""
        print(f"[STREAM] Starting snow stream processing...")
        
        self.snow_cap = cv2.VideoCapture(SNOW_CAMERA_RTSP, cv2.CAP_FFMPEG)
        if not self.snow_cap.isOpened():
            print(f"[STREAM] ERROR: Cannot open snow camera: {SNOW_CAMERA_RTSP}")
            return
        
        fail_count = 0
        MAX_FAILS = 15
        
        # Создаем окно для снеговой камеры (если включено)
        snow_window_created = False
        if SHOW_STREAM_WINDOW:
            try:
                cv2.namedWindow("Snow Camera - Line Crossing Detection", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Snow Camera - Line Crossing Detection", 1280, 720)
                snow_window_created = True
                print(f"[STREAM] ✅ Snow camera display window created")
            except Exception as e:
                print(f"[STREAM] ⚠️ WARNING: Failed to create snow display window: {e}")
        
        while not self._stop_event.is_set():
            ret, frame = self.snow_cap.read()
            
            if not ret or frame is None or frame.size == 0:
                fail_count += 1
                if fail_count >= MAX_FAILS:
                    print(f"[STREAM] Snow camera failed {MAX_FAILS} times, reconnecting...")
                    self.snow_cap.release()
                    time.sleep(2)
                    self.snow_cap = cv2.VideoCapture(SNOW_CAMERA_RTSP, cv2.CAP_FFMPEG)
                    fail_count = 0
                time.sleep(0.05)
                continue
            
            fail_count = 0
            
            # Сохраняем кадр в буфер для номерной камеры
            current_timestamp = time.time()
            timestamped_frame = TimestampedFrame(frame=frame.copy(), timestamp=current_timestamp)
            
            with self._snow_buffer_lock:
                # Удаляем старые кадры (старше 3 секунд)
                while len(self._snow_frame_buffer) > 0:
                    oldest = self._snow_frame_buffer[0]
                    if current_timestamp - oldest.timestamp > 3.0:
                        self._snow_frame_buffer.popleft()
                    else:
                        break
                
                self._snow_frame_buffer.append(timestamped_frame)
            
            # Детектируем грузовики на снеговой камере
            detections = self._detect_vehicles(frame)
            
            crossing_tracks = []
            if detections:
                # Обрабатываем кадр через детектор пересечения линии
                crossing_tracks = self.snow_detector.process_frame(frame, detections)
                
                # Обрабатываем пересечения (когда центр грузовика пересекает линию)
                for track in crossing_tracks:
                    if track.crossed:
                        # Проверяем, что центр грузовика пересек линию
                        h, w = frame.shape[:2]
                        line_y = int(h * self.snow_detector.line_y_ratio)
                        center_y = track.center[1]
                        
                        # Для снеговой камеры: захватываем кадр когда центр пересекает линию
                        if (self.snow_detector.direction == "down" and center_y >= line_y) or \
                           (self.snow_detector.direction == "up" and center_y <= line_y):
                            print(f"[STREAM] Snow camera: center crossing detected: track_id={track.track_id}, center_y={center_y}, line_y={line_y}")
                            # Сохраняем кадр в буфер для использования при пересечении на номерной камере
                            with self._snow_crossing_lock:
                                self._snow_crossing_frames.append({
                                    "frame": frame.copy(),
                                    "timestamp": current_timestamp,
                                    "track_id": track.track_id
                                })
            
            # Отображаем снеговой кадр с визуализацией (если включено)
            if SHOW_STREAM_WINDOW and snow_window_created:
                try:
                    display_snow_frame = self._draw_snow_line_and_tracks(frame, detections, crossing_tracks)
                    cv2.imshow("Snow Camera - Line Crossing Detection", display_snow_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:  # 'q' или ESC для выхода
                        self._stop_event.set()
                        break
                except Exception as e:
                    if "window" in str(e).lower() or "destroyed" in str(e).lower():
                        snow_window_created = False
                    else:
                        print(f"[STREAM] Error displaying snow frame: {e}")
            
            time.sleep(0.01)  # Небольшая пауза
        
        if SHOW_STREAM_WINDOW and snow_window_created:
            try:
                cv2.destroyWindow("Snow Camera - Line Crossing Detection")
            except Exception:
                pass
        
        if self.snow_cap is not None:
            self.snow_cap.release()
        print(f"[STREAM] Snow processing loop stopped")
    
    def _get_snow_frame(self, prefer_crossing: bool = True) -> Optional[np.ndarray]:
        """
        Получает кадр из снегового буфера.
        Если prefer_crossing=True, пытается взять кадр из буфера пересечений (когда центр пересек линию).
        Иначе берет последний кадр из общего буфера.
        """
        # Сначала пытаемся взять кадр из буфера пересечений (когда центр пересек линию)
        if prefer_crossing:
            with self._snow_crossing_lock:
                if len(self._snow_crossing_frames) > 0:
                    # Берем последний кадр пересечения
                    crossing_data = self._snow_crossing_frames[-1]
                    # Удаляем старые кадры (старше 2 секунд)
                    current_time = time.time()
                    while len(self._snow_crossing_frames) > 0:
                        oldest = self._snow_crossing_frames[0]
                        if current_time - oldest["timestamp"] > 2.0:
                            self._snow_crossing_frames.popleft()
                        else:
                            break
                    print(f"[STREAM] Using snow frame from center crossing (track_id={crossing_data['track_id']})")
                    return crossing_data["frame"].copy()
        
        # Fallback: берем последний кадр из общего буфера
        with self._snow_buffer_lock:
            if len(self._snow_frame_buffer) == 0:
                return None
            # Возвращаем последний кадр
            return self._snow_frame_buffer[-1].frame.copy()
    
    def _encode_frame_to_jpeg(self, frame: np.ndarray) -> Optional[bytes]:
        """Кодирует кадр в JPEG"""
        try:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                return None
            return buf.tobytes()
        except Exception as e:
            print(f"[STREAM] Error encoding frame: {e}")
            return None
    
    async def _process_crossing(self, track: Track, plate_frame: np.ndarray):
        """Обрабатывает пересечение линии - отправляет кадры в Gemini"""
        try:
            current_time = time.time()
            
            # Получаем снеговой кадр (предпочитаем кадр когда центр пересек линию)
            snow_frame = self._get_snow_frame(prefer_crossing=True)
            if snow_frame is None:
                print(f"[STREAM] No snow frame available, skipping")
                return
            
            # Кодируем кадры в JPEG
            plate_bytes = self._encode_frame_to_jpeg(plate_frame)
            snow_bytes = self._encode_frame_to_jpeg(snow_frame)
            
            if plate_bytes is None or snow_bytes is None:
                print(f"[STREAM] Failed to encode frames, skipping")
                return
            
            print(f"[STREAM] Line crossing detected! Sending frames to Gemini: plate_size={len(plate_bytes)}, snow_size={len(snow_bytes)}")
            
            # Отправляем в Gemini для анализа номера и снега
            gemini_result = None
            try:
                gemini_result = await self.merger.analyze_with_gemini(
                    snow_photo=snow_bytes,
                    plate_photo_1=plate_bytes,
                    plate_photo_2=None,
                    camera_plate=None,  # Gemini сам распознает номер
                )
                print(f"[STREAM] Gemini analysis completed: {gemini_result}")
            except Exception as e:
                print(f"[STREAM] Error in Gemini analysis: {e}")
                import traceback
                print(f"[STREAM] Traceback: {traceback.format_exc()}")
                return
            
            # Формируем event_data на основе результатов Gemini
            now_utc = datetime.now(timezone.utc)
            now_iso = now_utc.isoformat().replace('+00:00', 'Z')
            
            # Номер от Gemini
            plate = gemini_result.get("plate") if gemini_result else None
            plate_confidence = gemini_result.get("plate_confidence", 0.0) if gemini_result else 0.0
            
            # Дедупликация по номеру от Gemini
            if plate:
                with self._plates_lock:
                    # Очищаем старые записи
                    plates_to_remove = [
                        p for p, ts in self._processed_plates.items()
                        if current_time - ts > DEDUP_WINDOW_SECONDS
                    ]
                    for p in plates_to_remove:
                        del self._processed_plates[p]
                    
                    # Проверяем дубликат
                    if plate in self._processed_plates:
                        print(f"[STREAM] Duplicate plate detected: {plate}, skipping")
                        return
                    
                    # Помечаем как обработанный
                    self._processed_plates[plate] = current_time
                    print(f"[STREAM] Plate recognized by Gemini: {plate} (confidence: {plate_confidence})")
            
            event_data = {
                "camera_id": os.getenv("PLATE_CAMERA_ID", "camera-001"),
                "event_time": now_iso,
                "plate": plate,
                "confidence": plate_confidence,
                "direction": self.plate_detector.direction,
                "lane": 0,
                "vehicle": {},
                "plate_source": "gemini",
                "snow_volume_percentage": gemini_result.get("snow_percentage", 0.0) if gemini_result else 0.0,
                "snow_volume_confidence": gemini_result.get("snow_confidence", 0.0) if gemini_result else 0.0,
                "matched_snow": True,
                "gemini_result": gemini_result,
                "timestamp": now_iso,
            }
            
            # Отправляем на upstream через merger (используем его метод отправки)
            # merger уже имеет логику отправки, но нам нужна прямая отправка
            # Используем httpx напрямую
            upstream_url = os.getenv(
                "UPSTREAM_URL",
                "https://snowops-anpr-service.onrender.com/api/v1/anpr/events"
            )
            
            if upstream_url:
                try:
                    import httpx
                    event_str = json.dumps(event_data, ensure_ascii=False)
                    data = {"event": event_str}
                    files = []
                    
                    if plate_bytes:
                        files.append(("photos", ("detectionPicture.jpg", plate_bytes, "image/jpeg")))
                    if snow_bytes:
                        files.append(("photos", ("snowSnapshot.jpg", snow_bytes, "image/jpeg")))
                    
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        resp = await client.post(upstream_url, data=data, files=files or None)
                        upstream_result = {
                            "sent": resp.is_success,
                            "status": resp.status_code,
                            "error": None if resp.is_success else resp.text[:400],
                        }
                        print(f"[STREAM] Upstream result: sent={upstream_result.get('sent')}, status={upstream_result.get('status')}")
                except Exception as e:
                    print(f"[STREAM] Error sending to upstream: {e}")
                    import traceback
                    print(f"[STREAM] Traceback: {traceback.format_exc()}")
            else:
                print(f"[STREAM] UPSTREAM_URL not set, skipping upstream send")
            
        except Exception as e:
            print(f"[STREAM] Error processing crossing: {e}")
            import traceback
            print(f"[STREAM] Traceback: {traceback.format_exc()}")
    
    def _draw_line_and_tracks(self, frame: np.ndarray, detections: list, crossing_tracks: list) -> np.ndarray:
        """Отрисовывает линию пересечения, детекции и треки на кадре номерной камеры"""
        if not SHOW_STREAM_WINDOW:
            return frame
        
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # Рисуем линию пересечения для номерной камеры (используем значения из детектора)
        line_y = int(h * self.plate_detector.line_y_ratio)
        color = (0, 255, 0) if self.plate_detector.direction == "down" else (0, 0, 255)
        cv2.line(display_frame, (0, line_y), (w, line_y), color, 2)
        cv2.putText(display_frame, f"Line ({self.plate_detector.direction})", (10, line_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Рисуем детекции
        for x1, y1, x2, y2, conf in detections:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(display_frame, f"{conf:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Рисуем треки и пересечения
        for track in self.plate_detector.tracks.values():
            x1, y1, x2, y2 = track.bbox
            center_x, center_y = track.center
            
            # Цвет трека: зеленый если пересек, красный если нет
            track_color = (0, 255, 0) if track.crossed else (0, 165, 255)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), track_color, 2)
            cv2.circle(display_frame, (center_x, center_y), 5, track_color, -1)
            cv2.putText(display_frame, f"ID:{track.track_id}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, track_color, 1)
            
            if track.crossed:
                cv2.putText(display_frame, "CROSSED!", (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return display_frame
    
    def _processing_loop(self):
        """Основной цикл обработки номерного потока"""
        print(f"[STREAM] Starting plate stream processing...")
        print(f"[STREAM] DEBUG: SHOW_STREAM_WINDOW={SHOW_STREAM_WINDOW}")
        
        self.plate_cap = cv2.VideoCapture(PLATE_CAMERA_RTSP, cv2.CAP_FFMPEG)
        if not self.plate_cap.isOpened():
            print(f"[STREAM] ERROR: Cannot open plate camera: {PLATE_CAMERA_RTSP}")
            return
        
        fail_count = 0
        MAX_FAILS = 15
        
        # Создаем окно для отображения (если включено)
        window_created = False
        print(f"[STREAM] DEBUG: Checking SHOW_STREAM_WINDOW={SHOW_STREAM_WINDOW} (type: {type(SHOW_STREAM_WINDOW)})")
        if SHOW_STREAM_WINDOW:
            try:
                print(f"[STREAM] Creating display window...")
                print(f"[STREAM] SHOW_STREAM_WINDOW={SHOW_STREAM_WINDOW}")
                # Создаем окно с явными параметрами
                cv2.namedWindow("Plate Camera - Line Crossing Detection", cv2.WINDOW_NORMAL)
                # Пробуем установить размер
                try:
                    cv2.resizeWindow("Plate Camera - Line Crossing Detection", 1280, 720)
                except Exception:
                    pass  # Игнорируем ошибку изменения размера
                window_created = True
                print(f"[STREAM] ✅ Display window created (press 'q' to quit)")
                # Показываем пустой кадр для инициализации окна
                import numpy as np
                dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(dummy_frame, "Initializing...", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("Plate Camera - Line Crossing Detection", dummy_frame)
                cv2.waitKey(1)  # Важно: вызываем waitKey для отображения окна
                print(f"[STREAM] Window should be visible now")
            except Exception as e:
                print(f"[STREAM] ⚠️ WARNING: Failed to create display window: {e}")
                import traceback
                print(f"[STREAM] Traceback: {traceback.format_exc()}")
                print(f"[STREAM] Continuing without display window...")
        
        # Создаем event loop для асинхронных операций
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while not self._stop_event.is_set():
            ret, frame = self.plate_cap.read()
            
            if not ret or frame is None or frame.size == 0:
                fail_count += 1
                if fail_count >= MAX_FAILS:
                    print(f"[STREAM] Plate camera failed {MAX_FAILS} times, reconnecting...")
                    self.plate_cap.release()
                    time.sleep(2)
                    self.plate_cap = cv2.VideoCapture(PLATE_CAMERA_RTSP, cv2.CAP_FFMPEG)
                    fail_count = 0
                time.sleep(0.05)
                continue
            
            fail_count = 0
            
            # Детектируем машины
            detections = self._detect_vehicles(frame)
            
            crossing_tracks = []
            if detections:
                # Обрабатываем кадр через детектор пересечения линии (номерной камеры)
                crossing_tracks = self.plate_detector.process_frame(frame, detections)
                
                # Обрабатываем пересечения
                for track in crossing_tracks:
                    if track.crossed:
                        print(f"[STREAM] Line crossing detected: track_id={track.track_id}, bbox={track.bbox}")
                        # Запускаем асинхронную обработку
                        loop.run_until_complete(self._process_crossing(track, frame))
            
            # Отображаем кадр с визуализацией (если включено)
            if SHOW_STREAM_WINDOW and window_created:
                try:
                    display_frame = self._draw_line_and_tracks(frame, detections, crossing_tracks)
                    cv2.imshow("Plate Camera - Line Crossing Detection", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:  # 'q' или ESC для выхода
                        print(f"[STREAM] Display window closed by user")
                        self._stop_event.set()
                        break
                except Exception as e:
                    # Если окно было закрыто или произошла ошибка
                    if "window" in str(e).lower() or "destroyed" in str(e).lower():
                        print(f"[STREAM] Display window was closed")
                        window_created = False
                    else:
                        print(f"[STREAM] Error displaying frame: {e}")
            
            time.sleep(0.01)  # Небольшая пауза
        
        loop.close()
        
        if SHOW_STREAM_WINDOW and window_created:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        
        if self.plate_cap is not None:
            self.plate_cap.release()
        print(f"[STREAM] Processing loop stopped")
    
    def start(self):
        """Запускает обработку потоков"""
        if self._processing_thread is not None:
            print(f"[STREAM] Already running")
            return
        
        self._stop_event.clear()
        
        # Запускаем обработку снегового потока (с детекцией)
        self._snow_processing_thread = threading.Thread(
            target=self._snow_processing_loop,
            daemon=True,
            name="snow-processor"
        )
        self._snow_processing_thread.start()
        
        # Даем время на инициализацию снегового потока
        time.sleep(2)
        
        # Запускаем основной цикл обработки номерной камеры
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
            name="plate-processor"
        )
        self._processing_thread.start()
        
        print(f"[STREAM] Stream processor started (both cameras)")
    
    def stop(self):
        """Останавливает обработку потоков"""
        self._stop_event.set()
        if self._processing_thread is not None:
            self._processing_thread.join(timeout=5)
            self._processing_thread = None
        print(f"[STREAM] Stream processor stopped")


# Глобальный экземпляр
_stream_processor: Optional[StreamProcessor] = None


def init_stream_processor(merger) -> StreamProcessor:
    """Инициализирует глобальный экземпляр StreamProcessor"""
    global _stream_processor
    if _stream_processor is None:
        _stream_processor = StreamProcessor(merger)
    return _stream_processor


def get_stream_processor() -> Optional[StreamProcessor]:
    """Возвращает глобальный экземпляр StreamProcessor"""
    return _stream_processor

