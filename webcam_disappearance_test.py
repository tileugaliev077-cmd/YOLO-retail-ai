"""
Быстрый тест: заметит ли YOLOv8n исчезновение конкретных предметов со стола/полки.

Как использовать:
  1. Разложите предметы (стакан, телефон, духи) в кадре веб-камеры.
  2. Запустите скрипт — он покажет, что модель видит ПРЯМО СЕЙЧАС (список классов).
  3. Уберите предметы по одному — в консоли появится сообщение об исчезновении.
  4. Нажмите ESC для выхода.

ЧЕСТНО: YOLOv8n обучен на датасете COCO (80 общих классов). Из типичного набора
"стакан / телефон / духи" стандартная модель видит:
  - стакан  -> класс "cup"          (обычно распознается уверенно)
  - телефон -> класс "cell phone"   (обычно распознается уверенно)
  - духи    -> НЕТ такого класса в COCO. В лучшем случае модель может по ошибке
               отнести флакон к классу "bottle", если он похож по форме, но чаще
               просто не заметит его как отдельный объект. Это ограничение модели,
               а не скрипта — для реального распознавания духов нужно дообучение
               на своих фото (transfer learning).
"""

import cv2
import time
from ultralytics import YOLO

# Какие классы COCO отслеживаем. Полный список классов модель печатает при старте.
WATCHED_CLASSES = {"cup", "cell phone", "bottle"}

CONF_THRESHOLD = 0.4
OCCLUSION_TOLERANCE = 3   # столько кадров подряд объект должен отсутствовать, прежде чем считать его пропавшим
CHECK_EVERY_N_FRAMES = 5  # не гоняем YOLO на каждом кадре — не нужно для этого теста


def main():
    print("Загружаю модель YOLOv8n...")
    model = YOLO("yolov8n.pt")
    print(f"Модель загружена. Отслеживаем классы: {WATCHED_CLASSES}\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть веб-камеру.")
        return

    present_classes = set()       # что видим прямо сейчас (подтверждено)
    missing_streak = {}           # счетчик подряд-пустых проверок на каждый класс
    frame_count = 0

    print("Нажмите ESC для выхода. Уберите предметы по одному, чтобы проверить детекцию исчезновения.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Потерян сигнал с камеры.")
            break

        frame_count += 1
        detected_now = set()

        if frame_count % CHECK_EVERY_N_FRAMES == 0:
            results = model.predict(frame, verbose=False, conf=CONF_THRESHOLD)
            result = results[0]

            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    conf = float(box.conf[0])

                    # Рисуем рамку для наглядности, даже если класс не в списке отслеживаемых —
                    # так виднее, что вообще видит модель в кадре
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = (0, 255, 0) if cls_name in WATCHED_CLASSES else (128, 128, 128)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    if cls_name in WATCHED_CLASSES:
                        detected_now.add(cls_name)

            # Обновляем счетчики отсутствия для каждого отслеживаемого класса
            for cls_name in WATCHED_CLASSES:
                if cls_name in detected_now:
                    missing_streak[cls_name] = 0
                    if cls_name not in present_classes:
                        print(f"✅ Обнаружено: '{cls_name}' появился в кадре.")
                        present_classes.add(cls_name)
                else:
                    if cls_name in present_classes:
                        missing_streak[cls_name] = missing_streak.get(cls_name, 0) + 1
                        if missing_streak[cls_name] >= OCCLUSION_TOLERANCE:
                            print(f"❌ ИСЧЕЗ: '{cls_name}' пропал из кадра "
                                  f"(в {time.strftime('%H:%M:%S')}).")
                            present_classes.discard(cls_name)
                            missing_streak[cls_name] = 0

        # Статус на экране
        status = f"В кадре: {', '.join(sorted(present_classes)) if present_classes else '(ничего из отслеживаемого)'}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Webcam Object Disappearance Test", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
