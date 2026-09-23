"""
Экспорт модели YOLO для ускоренного инференса на edge-устройствах.

ЧЕСТНО: этот скрипт готовит модель к экспорту, но реальный прирост FPS на конкретном
устройстве (Raspberry Pi 5 / Jetson Nano / Orange Pi) зависит от платы, охлаждения,
версии рантайма и разрешения ROI — цифры нужно замерять на месте через
benchmark_inference.py, а не считать заранее известными.

Использование:
    python export_model.py --format onnx
    python export_model.py --format openvino
    python export_model.py --format engine --device 0   # TensorRT, требует NVIDIA GPU/Jetson

Рекомендации по целевым платформам:
    - Raspberry Pi 5 (CPU, без TPU/NPU-ускорителя): формат "onnx" с onnxruntime,
      либо "ncnn" (Ultralytics поддерживает экспорт в ncnn — часто быстрее на ARM CPU).
    - Jetson Nano/Orin: формат "engine" (TensorRT) — требует установленного TensorRT
      непосредственно на самом устройстве Jetson, экспорт лучше делать прямо на нем,
      а не кросс-компилировать на другом железе.
    - Intel-based мини-ПК: формат "openvino" — дает заметный прирост именно на Intel CPU.

Квантование (INT8):
    INT8-квантование дает дополнительное ускорение, но требует калибровочного датасета —
    набора РЕАЛЬНЫХ кадров с ваших камер (не синтетики), иначе точность может заметно
    просесть на реальных условиях освещения магазина. Собери 100-300 кадров с ROI
    из реальной работы системы и передай их через --calib-dir.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def export(model_path, fmt, device, calib_dir):
    model = YOLO(model_path)

    export_kwargs = {"format": fmt}

    if fmt == "engine":
        # TensorRT: nms/половинная точность включаем для скорости на GPU/Jetson
        export_kwargs["device"] = device
        export_kwargs["half"] = True

    if fmt in ("onnx", "openvino") and calib_dir:
        # INT8-квантование с калибровкой на реальных кадрах (если каталог передан)
        calib_path = Path(calib_dir)
        if not calib_path.exists():
            print(f"⚠️ Каталог калибровки '{calib_dir}' не найден — экспорт без INT8-квантования.")
        else:
            export_kwargs["int8"] = True
            export_kwargs["data"] = str(calib_path)
            print(f"Используем калибровочные кадры из: {calib_dir}")

    print(f"Экспорт модели '{model_path}' в формат '{fmt}'...")
    exported_path = model.export(**export_kwargs)
    print(f"\n✅ Готово: {exported_path}")
    print("Теперь замерьте реальную скорость на целевом устройстве:")
    print(f"    python benchmark_inference.py --model {exported_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Экспорт YOLO-модели для edge-развертывания")
    parser.add_argument("--model", default="yolov8n.pt", help="Путь к исходной модели .pt")
    parser.add_argument("--format", required=True,
                         choices=["onnx", "openvino", "engine", "ncnn"],
                         help="Целевой формат экспорта")
    parser.add_argument("--device", default="0", help="GPU-устройство для TensorRT (только для --format engine)")
    parser.add_argument("--calib-dir", default=None,
                         help="Каталог с реальными калиброчными кадрами для INT8-квантования (опционально)")
    args = parser.parse_args()

    export(args.model, args.format, args.device, args.calib_dir)
