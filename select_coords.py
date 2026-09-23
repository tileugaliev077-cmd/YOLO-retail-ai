import cv2

camera_source = 0  # Индекс веб-камеры или RTSP-ссылка магазинной камеры

points = []  # Сюда накапливаем клики
paused_frame = None  # Кадр, "замороженный" для точного клика


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and paused_frame is not None:
        points.append((x, y))
        print(f"Точка {len(points)}: x={x}, y={y}")

        # Рисуем точку прямо на замороженном кадре для наглядности
        cv2.circle(paused_frame, (x, y), 5, (0, 0, 255), -1)

        # Если накопили ровно 2 точки — считаем это готовым ROI
        if len(points) == 2:
            (x1, y1), (x2, y2) = points
            roi = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            print(f"\nГотовый ROI для вставки в код: {roi}\n")
            cv2.rectangle(paused_frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)

        cv2.imshow("Select Coords", paused_frame)


cap = cv2.VideoCapture(camera_source)
if not cap.isOpened():
    print("Ошибка: не удалось подключиться к камере. Проверьте индекс/RTSP-ссылку.")
    exit()

cv2.namedWindow("Select Coords")
cv2.setMouseCallback("Select Coords", click_event)

print("Управление:")
print("  ПРОБЕЛ — поставить видео на паузу и начать выбор точек")
print("  ЛКМ    — кликнуть по углу полки (нужно 2 точки: левый верх и правый низ)")
print("  R      — сбросить точки и снова поставить на паузу")
print("  ESC    — выход\n")

while True:
    if paused_frame is None:
        ret, frame = cap.read()
        if not ret:
            print("Потерян сигнал с камеры.")
            break
        cv2.imshow("Select Coords", frame)

    key = cv2.waitKey(30) & 0xFF

    if key == 27:  # ESC
        break
    elif key == ord(' ') and paused_frame is None:
        # Замораживаем текущий кадр для точного клика
        paused_frame = frame.copy()
        points = []
        cv2.imshow("Select Coords", paused_frame)
        print("Видео на паузе. Кликните по двум углам полки.")
    elif key == ord('r') or key == ord('R'):
        # Сброс — возвращаемся к живому видео
        paused_frame = None
        points = []
        print("Сброшено. Снова живое видео, нажмите ПРОБЕЛ для новой попытки.")

cap.release()
cv2.destroyAllWindows()
