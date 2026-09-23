# Smart Retail Monitor

A shelf-monitoring prototype for retail stores: it detects activity at a shelf zone, cross-checks it against POS receipt data, and notifies a staff member for manual review — **it never accuses a customer automatically.**

> ⚠️ \*\*This is a pilot/educational prototype, not a finished commercial product.\*\* Read \[Limitations \& Risks](#limitations--risks) before deploying it in a real store.

## System Architecture

```mermaid
flowchart LR
    subgraph Cameras
        C1\[Camera 1] --> FG1\[FrameGrabber]
        C2\[Camera 2] --> FG2\[FrameGrabber]
        C3\[Camera N] --> FG3\[FrameGrabber]
    end

    FG1 --> Q\[(Shared Frame Queue)]
    FG2 --> Q
    FG3 --> Q

    Q --> IE\[InferenceEngine<br/>batched YOLO + CLAHE lighting normalization<br/>occlusion-tolerant state machine<br/>relocation detection via color signature]

    IE -->|zone stayed occupied past grace period| PE\[(pending\_events)]
    POS\[POS / Cash Register<br/>sales\_export.txt] --> AR\[ArbitratorThread]
    PE --> AR
    AR -->|receipt found| Closed\[Event closed]
    AR -->|no receipt after timeout| Alert\[Staff audit notification via Telegram]
```

* **`FrameGrabber`** — one thread per camera. Only handles capture and RTSP reconnect. No ML here.
* **`InferenceEngine`** — a single thread for the whole system. Collects a mini-batch of frames, runs YOLO once per batch, applies CLAHE lighting normalization, tracks per-zone occupancy state with occlusion tolerance, and detects shelf-to-shelf relocation via a lightweight color-histogram signature.
* **`ArbitratorThread`** — matches events against POS receipt data (`sales\_export.txt`) and decides whether to notify staff.

## Key Safety Principles

* The system **never** declares theft on its own. When the receipt wait times out, the event status is `audit\_sent` — a request for a staff member to review, not a public accusation.
* If an item is simply moved to another zone in the same camera's field of view, this is recognized via color-signature matching and **cancels** an already-raised alert if the receipt hasn't arrived yet.
* A single receipt cannot close two simultaneous events for the same item (protects against a race condition when several people take the same product at once).
* Brief occlusion (a customer's body or hand blocking the shelf for a moment) does **not** immediately count as "item gone" — the zone must be empty for several consecutive checks before a departure is registered.

## Installation

```bash
git clone <your-repo-url>
cd smart-retail-monitor
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
# fill in TELEGRAM\_BOT\_TOKEN and TELEGRAM\_CHAT\_ID in .env
```

### Docker

```bash
cp .env.example .env
# fill in .env, then:
docker compose up --build
```

`docker-compose.yml` mounts `sales\_export.txt`, `maintenance.flag` and `events/` as volumes so the POS export and maintenance toggle work without rebuilding the image. It uses `network\_mode: host` by default to simplify reaching RTSP cameras on the store's local network — adjust if your deployment needs isolated networking instead.

## Picking Shelf Zones (ROI)

```bash
python select\_coords.py
```

Space pauses the frame, click the two corners of the zone, `R` resets, `ESC` exits. The resulting `(x1, y1, x2, y2)` tuple is printed to the console — paste it into `cameras\_config` in `smart\_retail\_monitor.py`.

## Running

```bash
python smart\_retail\_monitor.py
```

Edit `cameras\_config` in the `if \_\_name\_\_ == "\_\_main\_\_":` block:

```python
cameras\_config = \[
    {"source": 0, "name": "Shelf 1 (Coffee)", "coords": (100, 100, 500, 400), "item": "Jacobs Coffee"},
    # {"source": "rtsp://admin:PASS@192.168.1.50:554/stream2", "name": "Shelf 2", "coords": (...), "item": "..."},
]
```

For RTSP cameras, connect to the **sub-stream** (lower resolution) if your camera/NVR provides one — this noticeably reduces CPU load during inference.

## POS Integration

`ArbitratorThread` reads `sales\_export.txt` line by line, format:

```
UNIX\_TIMESTAMP|ITEM\_NAME
1735000000.0|Jacobs Coffee
```

Set up your POS system (1C, UMAG, etc.) to write to this file, or replace `\_read\_new\_sales()` with a direct call to your POS API if one is available.

## Maintenance Mode (Restocking)

Create a `maintenance.flag` file while restocking shelves — the system keeps pulling frames but stops raising events or sending alerts:

```bash
touch maintenance.flag   # enable
rm maintenance.flag      # disable — all zone states reset automatically
```

The in-store monitor/NVR keeps showing live video regardless, since it's an independent RTSP client separate from this script.

## Edge Deployment (ONNX / OpenVINO / TensorRT)

```bash
python export\_model.py --format onnx        # general CPU acceleration
python export\_model.py --format openvino    # Intel CPU
python export\_model.py --format engine --device 0   # TensorRT, run ON the Jetson device itself
```

INT8 quantization needs a calibration set of **real frames from your own cameras** (pass `--calib-dir`) — calibrating on synthetic data can hurt accuracy under your store's actual lighting.

## Limitations \& Risks

Please read this before deploying:

* **Does not recognize specific products.** Uses stock YOLOv8n (COCO, 80 general classes) — the "this zone = this item" mapping is geometric (ROI-based), not visual product recognition. Real SKU recognition requires fine-tuning on your own product photos.
* **Shelf-to-shelf relocation detection is a heuristic, not a guarantee.** Color-histogram matching can confuse visually similar items and may fail to match the same item across different lighting/angles, especially between different physical cameras.
* **Occlusion tolerance is a debounce, not full multi-object tracking.** It waits a few consecutive empty checks before declaring an item gone, which handles brief occlusion (a body or hand blocking the view for a second) but is not a substitute for a full MOT (Multi-Object Tracking) pipeline if your use case has long, frequent occlusions.
* **Does not replace human judgment.** Every alert is a request for manual review, not a verdict. Restrict alert delivery (Telegram, etc.) to the staff actually responsible for reviewing it.
* **Hardware-bound performance.** On CPU-only hardware (e.g. a used Core i5 laptop), realistically expect 1-2 cameras in real time without noticeable lag; 5+ cameras typically need a more capable machine or GPU/edge accelerator. Always verify with `benchmark\_inference.py` on your actual target device before committing to a hardware budget.
* **Legal review required locally.** Using video surveillance in a retail space typically requires a visible notice to customers under local data-protection law — confirm the exact requirement with a lawyer in your jurisdiction; this project does not provide legal advice.

## Project Files

|File|Purpose|
|-|-|
|`smart\_retail\_monitor.py`|Main system: camera capture, batched YOLO inference, POS reconciliation|
|`select\_coords.py`|Mouse-click utility to pick ROI coordinates|
|`benchmark\_inference.py`|Measures real inference speed on your hardware|
|`export\_model.py`|Exports the model to ONNX / OpenVINO / TensorRT for edge deployment|
|`Dockerfile`, `docker-compose.yml`|Containerized deployment|
|`requirements.txt`|Python dependencies|
|`.env.example`|Environment variable template|



