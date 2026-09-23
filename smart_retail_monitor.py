"""
Smart Retail Monitor — версия с общей очередью кадров и батч-инференсом.

Архитектура:
  FrameGrabber (по потоку на камеру)  -->  общая очередь кадров  -->  InferenceEngine (один поток, батч YOLO)
                                                                              |
                                                                              v
                                                                     pending_events
                                                                              ^
                                                                              |
                                                              ArbitratorThread (сверка с кассой)

Почему так лучше, чем "по YOLO-модели на камеру":
  - Одна загруженная модель в памяти вместо N копий.
  - Инференс идет батчем (несколько кадров за один проход по сети) — существенно
    эффективнее, чем N последовательных вызовов на разных CPU-ядрах под GIL.
  - Логика "занята ли зона" вынесена в один процесс, что упрощает будущий переход на GPU
    (замена CPU на GPU — это одна точка в коде, а не N).
"""

import cv2
import time
import queue
import threading
import os
from dotenv import load_dotenv
from ultralytics import YOLO
import requests

load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ==========================================
# ОБЩИЕ СТРУКТУРЫ ДАННЫХ
# ==========================================
frame_queue = queue.Queue(maxsize=20)   # если инференс не успевает — старые кадры отбрасываются, а не копятся
pending_events = []
sold_events = []
lock = threading.Lock()

# Недавние "уходы" объектов из зон — используются, чтобы отличить перестановку на другую полку от кражи.
# Каждая запись: {"camera", "item", "signature", "time", "had_alert"}
recent_departures = []
RELOCATION_WINDOW = 60.0        # сколько секунд помним об "уходе" объекта, ожидая, не появится ли он в другой зоне
RELOCATION_SIMILARITY = 0.65    # порог схожести гистограмм (0..1, чем выше — тем строже совпадение)


def compute_signature(crop_bgr):
    """Дешевая 'визуальная сигнатура' объекта — цветовая гистограмма в HSV.
    Не является настоящей re-identification моделью: две вещи одного цвета
    могут быть спутаны, а один и тот же товар при резкой смене освещения — не опознан.
    Это эвристика, а не гарантия, но она практически бесплатна по ресурсам на CPU."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def signatures_match(sig_a, sig_b):
    if sig_a is None or sig_b is None:
        return 0.0
    return cv2.compareHist(sig_a, sig_b, cv2.HISTCMP_CORREL)


def send_security_audit_alert(camera_name, item_name, frame):
    """Нейтральное уведомление для проверки сотрудником. Система не обвиняет, а передает на верификацию."""
    print(f"\nℹ️ [АУДИТ - {camera_name}] Товар '{item_name}' не найден в чеке. Отправка на проверку...")
    img_path = f"audit_{int(time.time())}.jpg"
    cv2.imwrite(img_path, frame)

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID не заданы в .env — уведомление не отправлено.")
        if os.path.exists(img_path):
            os.remove(img_path)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "caption": (
            f"🔍 ТРЕБУЕТСЯ ПРОВЕРКА СТАРШИМ СМЕНЫ\n"
            f"Камера: {camera_name}\nПолка: {item_name}\n"
            f"Статус: Истекло время ожидания чека."
        ),
    }
    try:
        with open(img_path, "rb") as photo:
            requests.post(url, data=data, files={"photo": photo}, timeout=5)
    except Exception as e:
        print(f"Ошибка отправки в Telegram: {e}")
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)


# ==========================================
# 1. FRAME GRABBER — только захват кадров, никакого ML здесь
# ==========================================
class FrameGrabber(threading.Thread):
    """Один поток на камеру. Единственная задача — держать соединение живым и класть кадры в очередь."""

    def __init__(self, camera_source, name, zone_coords, item_name):
        super().__init__(daemon=True)
        self.camera_source = camera_source
        self.name = name
        self.zone_coords = zone_coords
        self.item_name = item_name
        self.running = True
        self.cap = cv2.VideoCapture(camera_source)
        if not self.cap.isOpened():
            print(f"❌ ОШИБКА: Не удалось подключиться к камере '{self.name}' ({camera_source})!")

    def run(self):
        consecutive_fails = 0
        while self.running:
            if not self.cap.isOpened():
                print(f"⚠️ Камера '{self.name}' недоступна. Реконнект...")
                self.cap.open(self.camera_source)
                time.sleep(3)
                continue

            ret, frame = self.cap.read()
            if not ret:
                consecutive_fails += 1
                if consecutive_fails > 30:
                    print(f"⚠️ 'Тихий' обрыв потока на камере '{self.name}'. Перезапуск...")
                    self.cap.release()
                    self.cap.open(self.camera_source)
                    consecutive_fails = 0
                time.sleep(0.1)
                continue
            consecutive_fails = 0

            # Кладем кадр в очередь. Если очередь переполнена — выбрасываем самый старый кадр
            # (лучше потерять один устаревший кадр, чем блокировать захват видео).
            payload = {
                "camera": self.name,
                "zone": self.zone_coords,
                "item": self.item_name,
                "frame": frame,
                "captured_at": time.time(),
            }
            try:
                frame_queue.put_nowait(payload)
            except queue.Full:
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
                frame_queue.put_nowait(payload)

            # Ограничиваем частоту захвата — детекции кражи не нужно 30 FPS
            time.sleep(0.1)

    def stop(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()


# ==========================================
# 2. INFERENCE ENGINE — один поток, батч-инференс, вся ML-логика здесь
# ==========================================
class InferenceEngine(threading.Thread):
    """
    Забирает кадры из общей очереди, накапливает мини-батч и прогоняет через YOLO одним вызовом.
    Ведет состояние (idle/pending/alerted) для каждой камеры отдельно, с допуском на кратковременное
    перекрытие объекта (occlusion) — например, покупатель на секунду закрыл собой полку.
    """

    def __init__(self, model_path="yolov8n.pt", batch_size=4, batch_timeout=0.2,
                 grace_period=10.0, area_threshold=1500, conf_threshold=0.4,
                 maintenance_flag_path="maintenance.flag",
                 occlusion_tolerance=3, use_clahe=True):
        super().__init__(daemon=True)
        self.running = True
        self.model = YOLO(model_path)
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.grace_period = grace_period
        self.area_threshold = area_threshold
        self.conf_threshold = conf_threshold
        self.maintenance_flag_path = maintenance_flag_path
        self._was_in_maintenance = False
        # Сколько ПОДРЯД пустых проверок зоны допускается, прежде чем считать объект
        # реально пропавшим. Решает проблему перекрытия: покупатель на 1-2 кадра
        # закрыл собой полку спиной/рукой — это не должно считаться "уходом" объекта.
        self.occlusion_tolerance = occlusion_tolerance
        self.use_clahe = use_clahe
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if use_clahe else None
        # Состояние на каждую камеру:
        # {"status", "timer", "last_crop", "missing_streak"}
        self.zone_state = {}

    def _normalize_lighting(self, frame_bgr):
        """CLAHE (адаптивное выравнивание контраста) по каналу яркости в LAB.
        Сглаживает резкие тени и блики, не 'ослепляя' модель пересветом/недосветом
        при смене освещения в течение дня. Дешево по CPU — работает только на ROI,
        а не на всем кадре."""
        if not self.use_clahe or frame_bgr.size == 0:
            return frame_bgr
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _collect_batch(self):
        """Собирает до batch_size кадров, но не ждет дольше batch_timeout секунд."""
        batch = []
        deadline = time.time() + self.batch_timeout
        while len(batch) < self.batch_size and time.time() < deadline:
            try:
                remaining = max(0.0, deadline - time.time())
                item = frame_queue.get(timeout=remaining)
                batch.append(item)
            except queue.Empty:
                break
        return batch

    def run(self):
        while self.running:
            batch = self._collect_batch()
            if not batch:
                continue

            in_maintenance = os.path.exists(self.maintenance_flag_path)
            if in_maintenance != self._was_in_maintenance:
                if in_maintenance:
                    print("\n🛠️  РЕЖИМ ОБСЛУЖИВАНИЯ ВКЛЮЧЕН — выкладка товара, тревоги временно отключены "
                          "(видео с камер по-прежнему доступно на мониторе/NVR).")
                else:
                    print("\n✅ Режим обслуживания выключен — мониторинг возобновлен, состояния зон сброшены.")
                    for state in self.zone_state.values():
                        state["status"] = "idle"
                        state["timer"] = 0
                        state["last_crop"] = None
                        state["missing_streak"] = 0
                self._was_in_maintenance = in_maintenance

            if in_maintenance:
                # В режиме обслуживания просто "проглатываем" кадры, не запуская инференс и не создавая события —
                # это дешевле, чем гонять YOLO вхолостую на кадрах, где заведомо идет выкладка товара.
                continue

            # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: обрезаем кадр до ROI ДО инференса, а не после.
            # YOLO обрабатывает только пиксели полки, а не весь кадр камеры —
            # это реально сокращает объем вычислений, а не просто фильтрует результат.
            cropped_frames = []
            for item in batch:
                x1, y1, x2, y2 = item["zone"]
                cropped = item["frame"][y1:y2, x1:x2]
                cropped = self._normalize_lighting(cropped)
                cropped_frames.append(cropped)

            # Один вызов модели на весь батч обрезанных изображений
            results = self.model.predict(cropped_frames, verbose=False, conf=self.conf_threshold)

            for item, cropped, result in zip(batch, cropped_frames, results):
                self._process_result(item, cropped, result)

    def _process_result(self, item, cropped_frame, result):
        cam_name = item["camera"]
        now = time.time()

        # Кадр уже обрезан до границ ROI, поэтому любой найденный объект по определению
        # находится внутри зоны — дополнительная проверка координат больше не нужна.
        object_in_zone = result.boxes is not None and len(result.boxes) > 0

        state = self.zone_state.setdefault(
            cam_name, {"status": "idle", "timer": 0, "last_crop": None, "missing_streak": 0}
        )
        was_idle = state["status"] == "idle"

        if object_in_zone:
            state["missing_streak"] = 0
            state["last_crop"] = cropped_frame  # запоминаем последний вид объекта, пока он еще в зоне

            if was_idle:
                # МОМЕНТ ПОЯВЛЕНИЯ: проверяем, не тот ли это товар, что недавно пропал из другой зоны
                self._check_relocation_arrival(cam_name, item["item"], cropped_frame, now)
                state["status"] = "pending"
                state["timer"] = now
            elif state["status"] == "pending":
                if now - state["timer"] > self.grace_period:
                    with lock:
                        pending_events.append({
                            "camera": cam_name,
                            "item": item["item"],
                            "time": now,
                            "frame": item["frame"].copy(),
                            "status": "waiting_receipt",
                        })
                    state["status"] = "alerted"
            # status == "alerted" — просто держим объект в поле зрения дальше, ничего не меняем
        else:
            if was_idle:
                return  # и так пусто, нечего обрабатывать

            # ЗАЩИТА ОТ ПЕРЕКРЫТИЯ (occlusion): не сбрасываем состояние по первому же пустому кадру.
            # Человек, на секунду закрывший собой полку спиной или рукой, не должен считаться
            # "уходом" объекта — таймер grace_period продолжает идти, как будто объект все еще там.
            state["missing_streak"] += 1
            if state["missing_streak"] < self.occlusion_tolerance:
                return  # вероятно, временное перекрытие — ждем еще несколько проверок

            # Порог перекрытия превышен — считаем это настоящим уходом объекта из зоны
            self._register_departure(cam_name, item["item"], state["last_crop"], now,
                                      had_alert=(state["status"] == "alerted"))
            state["status"] = "idle"
            state["timer"] = 0
            state["last_crop"] = None
            state["missing_streak"] = 0

    def _register_departure(self, camera, item_name, last_crop, ts, had_alert):
        signature = compute_signature(last_crop)
        if signature is None:
            return
        with lock:
            recent_departures.append({
                "camera": camera,
                "item": item_name,
                "signature": signature,
                "time": ts,
                "had_alert": had_alert,
            })

    def _check_relocation_arrival(self, camera, item_name, arriving_crop, ts):
        signature = compute_signature(arriving_crop)
        if signature is None:
            return

        with lock:
            # Чистим устаревшие записи об "уходах" — вне окна релевантности они больше не в счет
            recent_departures[:] = [d for d in recent_departures if ts - d["time"] <= RELOCATION_WINDOW]

            best_match, best_score = None, 0.0
            for departure in recent_departures:
                if departure["camera"] == camera:
                    continue  # сравниваем только с другими зонами, не с самим собой
                score = signatures_match(signature, departure["signature"])
                if score > best_score:
                    best_score, best_match = score, departure

            if best_match and best_score >= RELOCATION_SIMILARITY:
                print(
                    f"🔄 Похоже на перестановку: товар с '{best_match['camera']}' "
                    f"обнаружен на '{camera}' (сходство {best_score:.2f}). Кражей не считается."
                )
                recent_departures.remove(best_match)

                # Сбрасываем состояние исходной зоны, чтобы она не создала собственную тревогу позже
                dep_state = self.zone_state.get(best_match["camera"])
                if dep_state:
                    dep_state["status"] = "idle"
                    dep_state["timer"] = 0

                # Если тревога по исходной зоне уже была создана, но чек еще не подтвердил её — отменяем
                if best_match["had_alert"]:
                    still_pending = [
                        e for e in pending_events
                        if e["camera"] == best_match["camera"]
                        and e["item"] == best_match["item"]
                        and e["status"] == "waiting_receipt"
                    ]
                    if still_pending:
                        still_pending[0]["status"] = "closed"
                        print(f"   ↳ Ранее созданная тревога по '{best_match['camera']}' отменена как ложная.")
                    else:
                        print(f"   ↳ ВНИМАНИЕ: тревога по '{best_match['camera']}' уже могла уйти в Telegram "
                              f"до обнаружения перестановки — отозвать отправленное сообщение нельзя, "
                              f"стоит вручную пометить как ложное сотруднику.")

    def stop(self):
        self.running = False


# ==========================================
# 3. АРБИТР — сверка событий с чеками (с защитой от повторного использования одного чека)
# ==========================================
class ArbitratorThread(threading.Thread):
    def __init__(self, sales_log_path="sales_export.txt", max_checkout_time=300):
        super().__init__(daemon=True)
        self.running = True
        self.sales_log_path = sales_log_path
        self.last_file_position = 0
        self.max_checkout_time = max_checkout_time

    def _read_new_sales(self):
        """Формат строки в файле выгрузки кассы: ВРЕМЯ_UNIX | НАЗВАНИЕ_ТОВАРА"""
        if not os.path.exists(self.sales_log_path):
            return
        try:
            with open(self.sales_log_path, "r", encoding="utf-8") as f:
                f.seek(self.last_file_position)
                lines = f.readlines()
                self.last_file_position = f.tell()
                for line in lines:
                    parts = line.strip().split("|")
                    if len(parts) >= 2:
                        sale_time = float(parts[0].strip())
                        item_name = parts[1].strip()
                        with lock:
                            sold_events.append({"time": sale_time, "item": item_name})
        except Exception as e:
            print(f"Ошибка чтения файла чеков кассы: {e}")

    def run(self):
        while self.running:
            time.sleep(2)
            self._read_new_sales()
            current_time = time.time()

            with lock:
                for event in pending_events:
                    if event["status"] != "waiting_receipt":
                        continue

                    # ВАЖНО: чек "потребляется" при сопоставлении и удаляется из sold_events,
                    # чтобы один и тот же чек не мог закрыть два разных события одновременно
                    # (проблема из предыдущего разбора: 3-4 одновременных изъятия одного товара).
                    matched_sale = None
                    for sold in sold_events:
                        if sold["item"] == event["item"] and sold["time"] >= event["time"]:
                            matched_sale = sold
                            break

                    if matched_sale:
                        sold_events.remove(matched_sale)
                        print(f"✅ Товар '{event['item']}' подтвержден чеком с кассы.")
                        event["status"] = "closed"
                    elif current_time - event["time"] > self.max_checkout_time:
                        event["status"] = "audit_sent"
                        send_security_audit_alert(event["camera"], event["item"], event["frame"])

                pending_events[:] = [e for e in pending_events if e["status"] == "waiting_receipt"]

    def stop(self):
        self.running = False


# ==========================================
# ТОЧКА ВХОДА
# ==========================================
if __name__ == "__main__":
    cameras_config = [
        {"source": 0, "name": "Полка №1 (Кофе)", "coords": (100, 100, 500, 400), "item": "Кофе Jacobs"},
        # Добавляйте сюда остальные камеры — все они пишут в один frame_queue,
        # и обрабатываются одним InferenceEngine, а не своим YOLO на каждую.
    ]

    grabbers = [FrameGrabber(c["source"], c["name"], c["coords"], c["item"]) for c in cameras_config]
    engine = InferenceEngine(batch_size=max(4, len(cameras_config)))
    arbitrator = ArbitratorThread(sales_log_path="sales_export.txt")

    all_threads = grabbers + [engine, arbitrator]

    for t in all_threads:
        t.start()

    print("\n🚀 Система запущена: общая очередь кадров + батч-инференс + сверка с кассой.")
    print("Для выхода нажмите Ctrl+C.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Завершение работы...")
        for t in all_threads:
            t.stop()
            t.join(timeout=2.0)
        print("Система успешно выключена.")
