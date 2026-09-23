"""
Бенчмарк скорости YOLO-инференса на вашем железе.

Замеряет реальное время обработки одного кадра и батча кадров, чтобы ответить
на практический вопрос "потянет ли этот компьютер N камер в реальном времени",
не полагаясь на общие оценки из документации.

Запуск:
    python benchmark_inference.py
    python benchmark_inference.py --cameras 5 --roi-size 400x300
"""

import argparse
import time
import numpy as np
from ultralytics import YOLO


def make_dummy_frame(width, height):
    """Синтетический кадр вместо реальной камеры — чтобы бенчмарк можно было
    запустить где угодно, без подключенных камер."""
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def maybe_save_chart(camera_counts, fps_values, output_path):
    """Сохраняет график FPS от числа камер — ТОЛЬКО по реально замеренным значениям
    на этом запуске. Никаких заглушек или примерных цифр — если данных нет, график
    не рисуется."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n⚠️ matplotlib не установлен (pip install matplotlib) — график не сохранен, "
              "но числовые результаты выше уже реальны и их можно использовать.")
        return

    plt.figure(figsize=(7, 4))
    plt.plot(camera_counts, fps_values, marker="o")
    plt.xlabel("Количество камер в батче")
    plt.ylabel("FPS (суммарно по батчу)")
    plt.title("Реально замеренная производительность на этом устройстве")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\n📊 График сохранен: {output_path} (по данным именно этого прогона на этом железе)")


def run_benchmark(model_path, roi_width, roi_height, n_cameras, warmup=5, iterations=30):
    print(f"Модель: {model_path}")
    print(f"Размер зоны (ROI): {roi_width}x{roi_height}")
    print(f"Имитация количества камер в батче: {n_cameras}\n")

    model = YOLO(model_path)

    dummy_frame = make_dummy_frame(roi_width, roi_height)
    batch = [dummy_frame.copy() for _ in range(n_cameras)]

    print("Прогрев модели (не учитывается в замере)...")
    for _ in range(warmup):
        model.predict(batch, verbose=False)

    print(f"Замер: {iterations} итераций...\n")
    single_times = []
    batch_times = []

    for _ in range(iterations):
        # Один кадр — типичная нагрузка при 1 камере
        t0 = time.perf_counter()
        model.predict([dummy_frame], verbose=False)
        single_times.append(time.perf_counter() - t0)

        # Батч — типичная нагрузка при n_cameras камерах одновременно
        t0 = time.perf_counter()
        model.predict(batch, verbose=False)
        batch_times.append(time.perf_counter() - t0)

    avg_single = sum(single_times) / len(single_times)
    avg_batch = sum(batch_times) / len(batch_times)
    per_frame_in_batch = avg_batch / n_cameras

    print("=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)
    print(f"Один кадр (1 камера):           {avg_single*1000:.1f} мс  (~{1/avg_single:.1f} FPS)")
    print(f"Батч из {n_cameras} кадров:              {avg_batch*1000:.1f} мс  "
          f"(~{n_cameras/avg_batch:.1f} FPS суммарно)")
    print(f"В среднем на кадр внутри батча:  {per_frame_in_batch*1000:.1f} мс")
    print()

    # Практический вывод относительно grace_period по умолчанию в проекте (10 сек)
    grace_period = 10.0
    checks_per_grace_period = grace_period / avg_batch if avg_batch > 0 else float("inf")
    print(f"При grace_period={grace_period}с и {n_cameras} камерах зона будет "
          f"проверяться ~{checks_per_grace_period:.0f} раз(а) за это время.")
    if checks_per_grace_period < 5:
        print("⚠️  Меньше 5 проверок за grace_period — рискованно мало для надежной детекции.")
        print("    Рекомендации: уменьшить разрешение ROI, снизить количество камер в одном")
        print("    процессе, использовать OpenVINO/ONNX-экспорт модели, либо добавить GPU.")
    else:
        print("✅ Запас проверок за grace_period выглядит достаточным.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Бенчмарк YOLO-инференса для Smart Retail Monitor")
    parser.add_argument("--model", default="yolov8n.pt", help="Путь к модели YOLO")
    parser.add_argument("--cameras", type=int, default=1, help="Сколько камер имитировать в батче")
    parser.add_argument("--roi-size", default="400x300", help="Размер зоны ROI в формате ШИРИНАxВЫСОТА")
    parser.add_argument("--iterations", type=int, default=30, help="Число замеров для усреднения")
    parser.add_argument("--sweep", action="store_true",
                         help="Прогнать замер для 1..--cameras камер и построить график (нужен matplotlib)")
    parser.add_argument("--chart-output", default="benchmark_result.png",
                         help="Куда сохранить график при --sweep")
    args = parser.parse_args()

    width, height = map(int, args.roi_size.lower().split("x"))

    if args.sweep:
        camera_counts, fps_values = [], []
        for n in range(1, args.cameras + 1):
            print(f"\n### {n} камер(а) ###")
            model = YOLO(args.model)
            dummy = make_dummy_frame(width, height)
            batch = [dummy.copy() for _ in range(n)]
            for _ in range(5):
                model.predict(batch, verbose=False)
            times = []
            for _ in range(args.iterations):
                t0 = time.perf_counter()
                model.predict(batch, verbose=False)
                times.append(time.perf_counter() - t0)
            avg = sum(times) / len(times)
            fps = n / avg
            print(f"Среднее время батча: {avg*1000:.1f} мс  →  {fps:.1f} FPS суммарно")
            camera_counts.append(n)
            fps_values.append(fps)
        maybe_save_chart(camera_counts, fps_values, args.chart_output)
    else:
        run_benchmark(args.model, width, height, args.cameras, iterations=args.iterations)
