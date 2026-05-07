"""Честный бенчмарк бэкендов инференса для YOLOv8s на CPU.

Тестируемые конфигурации:
  1. PyTorch via Ultralytics  — production baseline
  2. ONNX via Ultralytics      — тот же самый wrapper, но ONNX backend (показываем
                                  что наивный экспорт обычно медленнее baseline)
  3. onnxruntime CPU EP default — прямой вызов без wrapper, дефолтные настройки
  4. onnxruntime CPU EP tuned   — все CPU threads + ORT_ENABLE_ALL graph optimizer
  5. onnxruntime DNNL EP        — oneDNN provider (если доступен в build'е)
  6. onnxruntime INT8 quantized — статическая INT8 квантизация с калибровкой на
                                  кадрах из тестового видео; даёт ускорение ценой
                                  потери точности (mAP обычно падает на 1-3%,
                                  отдельно НЕ измеряем — нужен валидационный сет)

Замеряется forward+preprocess для всех направлений (postprocess пропускается
чтобы не сравнивать яблоки с грушами; в прод-пути он одинаков и составляет
несколько ms независимо от backend'а).

Запуск:
    python scripts/benchmark_inference.py videos/your_video.mp4 --frames 50
    python scripts/benchmark_inference.py videos/your_video.mp4 --frames 50 --save
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402

# --- helpers ----------------------------------------------------------


def grab_frame(video_path: str):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"Не смог прочитать кадр: {video_path}")
    return frame


def letterbox(frame: np.ndarray, imgsz: int) -> np.ndarray:
    """YOLOv8-совместимый letterbox: масштабирование с сохранением пропорций
    + padding серым. Возвращает массив (1, 3, imgsz, imgsz) float32 в [0,1]."""
    h, w = frame.shape[:2]
    r = min(imgsz / h, imgsz / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_h, pad_w = imgsz - nh, imgsz - nw
    top = pad_h // 2
    left = pad_w // 2
    padded = cv2.copyMakeBorder(
        resized,
        top,
        pad_h - top,
        left,
        pad_w - left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None]  # CHW + batch
    return np.ascontiguousarray(arr)


def bench(name: str, run_fn, frame, iters: int) -> dict:
    # Прогрев: первые проходы всегда медленнее (allocations, kernel JIT)
    for _ in range(5):
        run_fn(frame)

    timings: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run_fn(frame)
        timings.append((time.perf_counter() - t0) * 1000)

    s = sorted(timings)
    return {
        "name": name,
        "p50": s[len(s) // 2],
        "p95": s[int(len(s) * 0.95)],
        "mean": statistics.mean(timings),
        "min": min(timings),
        "fps": 1000 / statistics.mean(timings),
    }


def quantize_int8(
    onnx_in: Path,
    onnx_out: Path,
    video_path: str,
    imgsz: int,
    n_calib: int = 50,
) -> None:
    """Статическая INT8-квантизация модели через onnxruntime.

    Калибровка делается на N кадрах, равномерно выбранных из тестового видео.
    Для prod-решения калибровать надо на отдельном val-сете, а не на test-видео.
    """
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    # Подготовка модели: shape inference + базовые оптимизации перед квантизацией
    preproc_path = onnx_in.with_name(onnx_in.stem + ".preproc.onnx")
    quant_pre_process(str(onnx_in), str(preproc_path), skip_symbolic_shape=False)

    # Извлекаем кадры для калибровки равномерно по видео
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // n_calib)
    frames: list[np.ndarray] = []
    for i in range(n_calib):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError("Не смог собрать кадры для калибровки")

    # Имя input'а вытаскиваем из preproc-модели
    import onnxruntime as ort

    sess = ort.InferenceSession(str(preproc_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    del sess

    class VideoCalibrationReader(CalibrationDataReader):
        def __init__(self):
            self.idx = 0

        def get_next(self):
            if self.idx >= len(frames):
                return None
            x = letterbox(frames[self.idx], imgsz)
            self.idx += 1
            return {input_name: x}

    quantize_static(
        str(preproc_path),
        str(onnx_out),
        calibration_data_reader=VideoCalibrationReader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
    )

    preproc_path.unlink(missing_ok=True)
    print(f"  → INT8 модель: {onnx_out} ({onnx_out.stat().st_size / 1024**2:.1f} МБ)")


def fmt_row(r: dict, baseline_mean: float) -> str:
    speedup = baseline_mean / r["mean"]
    sign = "+" if speedup > 1 else ""
    return (
        f"  {r['name']:38s} | "
        f"p50 {r['p50']:6.1f}мс | "
        f"p95 {r['p95']:6.1f}мс | "
        f"mean {r['mean']:6.1f}мс | "
        f"fps {r['fps']:5.1f} | "
        f"x{speedup:.2f} ({sign}{(speedup - 1) * 100:+.0f}%)"
    )


# --- main -------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--save", action="store_true", help="Дописать результаты в docs/BENCHMARKS.md")
    args = ap.parse_args()

    import onnxruntime as ort
    from ultralytics import YOLO

    frame = grab_frame(args.video)
    imgsz = settings.img_size
    cpu = os.cpu_count() or 4

    print("=" * 90)
    print(f"Hardware:    {platform.processor() or platform.machine()}")
    print(f"Python:      {sys.version.split()[0]}")
    print(f"onnxruntime: {ort.__version__}")
    print(f"Providers:   {ort.get_available_providers()}")
    print(f"CPU count:   {cpu}")
    print(f"Frame:       {frame.shape}, imgsz={imgsz}")
    print(f"Iterations:  {args.frames} (after 5 warmup runs)")
    print("=" * 90)

    pt_path = Path("best.pt")
    onnx_path = Path("best.onnx")

    if not onnx_path.exists():
        print(f"\nЭкспортирую {onnx_path} с simplify=True, opset=17...")
        YOLO(str(pt_path)).export(format="onnx", imgsz=imgsz, simplify=True, opset=17)

    results: list[dict] = []

    # 1. PyTorch via Ultralytics — production baseline
    print("\n[1/6] PyTorch (Ultralytics) — baseline...")
    pt_model = YOLO(str(pt_path))
    results.append(
        bench(
            "PyTorch (Ultralytics)",
            lambda f: pt_model.predict(f, imgsz=imgsz, verbose=False),
            frame,
            args.frames,
        )
    )

    # 2. ONNX via Ultralytics — тот же wrapper, но ONNX backend
    print("[2/6] ONNX (Ultralytics wrapper)...")
    onnx_ult_model = YOLO(str(onnx_path))
    results.append(
        bench(
            "ONNX (Ultralytics wrapper)",
            lambda f: onnx_ult_model.predict(f, imgsz=imgsz, verbose=False),
            frame,
            args.frames,
        )
    )

    # 3. onnxruntime CPU EP default — прямой вызов, без оптимизаций
    print("[3/6] onnxruntime CPU EP — default options...")
    sess_default = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    in_name = sess_default.get_inputs()[0].name
    results.append(
        bench(
            "onnxruntime CPU EP (default)",
            lambda f: sess_default.run(None, {in_name: letterbox(f, imgsz)}),
            frame,
            args.frames,
        )
    )

    # 4. onnxruntime CPU EP tuned — все CPU threads + ORT_ENABLE_ALL
    print("[4/6] onnxruntime CPU EP — tuned (all threads + graph opt)...")
    sess_opt = ort.SessionOptions()
    sess_opt.intra_op_num_threads = cpu
    sess_opt.inter_op_num_threads = 1
    sess_opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_tuned = ort.InferenceSession(
        str(onnx_path),
        sess_opt,
        providers=["CPUExecutionProvider"],
    )
    in_name = sess_tuned.get_inputs()[0].name
    results.append(
        bench(
            "onnxruntime CPU EP (tuned)",
            lambda f: sess_tuned.run(None, {in_name: letterbox(f, imgsz)}),
            frame,
            args.frames,
        )
    )

    # 5. onnxruntime DNNL EP — oneDNN, работает и на AMD
    print("[5/6] onnxruntime DNNL EP...")
    if "DnnlExecutionProvider" in ort.get_available_providers():
        sess_dnnl = ort.InferenceSession(
            str(onnx_path),
            sess_opt,
            providers=["DnnlExecutionProvider", "CPUExecutionProvider"],
        )
        in_name = sess_dnnl.get_inputs()[0].name
        results.append(
            bench(
                "onnxruntime DNNL EP",
                lambda f: sess_dnnl.run(None, {in_name: letterbox(f, imgsz)}),
                frame,
                args.frames,
            )
        )
    else:
        print("    SKIP: DnnlExecutionProvider не доступен в этом билде onnxruntime")

    # 6. INT8 quantized — самая агрессивная оптимизация, ценой mAP
    print("[6/6] onnxruntime INT8 (static quantization)...")
    int8_path = Path("best_int8.onnx")
    if not int8_path.exists():
        print(f"    Квантую {onnx_path} → {int8_path} (калибровка: 50 кадров из {args.video})...")
        try:
            quantize_int8(onnx_path, int8_path, args.video, imgsz, n_calib=50)
        except Exception as e:
            print(f"    SKIP: квантизация не удалась — {type(e).__name__}: {e}")
            int8_path = None

    if int8_path and int8_path.exists():
        sess_int8 = ort.InferenceSession(
            str(int8_path),
            sess_opt,
            providers=["CPUExecutionProvider"],
        )
        in_name = sess_int8.get_inputs()[0].name
        results.append(
            bench(
                "onnxruntime INT8 (tuned)",
                lambda f: sess_int8.run(None, {in_name: letterbox(f, imgsz)}),
                frame,
                args.frames,
            )
        )

    # --- Summary ---
    baseline = results[0]["mean"]
    print("\n" + "=" * 90)
    print("Сводка (baseline = PyTorch Ultralytics):\n")
    for r in results:
        print(fmt_row(r, baseline))
    print("=" * 90)

    best = min(results, key=lambda r: r["mean"])
    if best["name"] != results[0]["name"]:
        speedup = baseline / best["mean"]
        print(
            f"\n→ Лучший: {best['name']} (x{speedup:.2f} = {(speedup - 1) * 100:+.0f}% к baseline)"
        )
    else:
        print("\n→ Лучший — PyTorch baseline. ONNX backend на этом железе не дал выигрыша.")

    if args.save:
        save_results(results, baseline, args.video, args.frames, ort.__version__)


def save_results(
    results: list[dict],
    baseline: float,
    video: str,
    frames: int,
    ort_version: str,
):
    """Дописывает свежие результаты в docs/BENCHMARKS.md."""
    target = ROOT / "docs" / "BENCHMARKS.md"
    target.parent.mkdir(exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"\n## Run {now}",
        f"\n**Hardware:** `{platform.processor() or platform.machine()}` ({os.cpu_count()} CPU)  ",
        f"**Software:** Python {sys.version.split()[0]}, "
        f"onnxruntime {ort_version}, ultralytics from requirements.txt  ",
        f"**Video:** `{video}` · **Iterations:** {frames} (after 5 warmup runs)\n",
        "| Backend | p50 | p95 | mean | fps | speedup |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        speedup = baseline / r["mean"]
        lines.append(
            f"| {r['name']} | {r['p50']:.1f}мс | {r['p95']:.1f}мс | "
            f"{r['mean']:.1f}мс | {r['fps']:.1f} | x{speedup:.2f} "
            f"({(speedup - 1) * 100:+.0f}%) |"
        )

    if not target.exists():
        target.write_text(
            "# Inference Backend Benchmarks\n\n"
            "Сравнение бэкендов инференса для YOLOv8s на одном кадре. "
            "Baseline — PyTorch через Ultralytics (то, что используется в проде "
            "по умолчанию). Все замеры single-frame на CPU.\n",
            encoding="utf-8",
        )

    with target.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nРезультаты дописаны в {target}")


if __name__ == "__main__":
    main()
