import os
import sys
import uuid
from typing import List, Tuple

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import subprocess

# Absolute paths to existing homework directories
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HW2_P3_DIR = os.path.join(WORKSPACE_ROOT, "HW2", "Part-3")
HW3_DIR = os.path.join(WORKSPACE_ROOT, "HW3")
HW4_WEBAPP_DIR = os.path.join(WORKSPACE_ROOT, "HW4", "Webapp")
HW7_PART1_DIR = os.path.join(WORKSPACE_ROOT, "hw7", "part1")
HW7_PART3_DIR = os.path.join(WORKSPACE_ROOT, "hw7", "part3")
HW5_DIR = os.path.join(WORKSPACE_ROOT, "HW5")

# Make external homework modules importable
if HW3_DIR not in sys.path:
    sys.path.append(HW3_DIR)
if HW4_WEBAPP_DIR not in sys.path:
    sys.path.append(HW4_WEBAPP_DIR)

# HW4 pure-function modules
try:
    from part1_stitching import run_panorama_stitching  # type: ignore
    from part2_sift import run_sift_comparison  # type: ignore
    HW4_AVAILABLE = True
except Exception:
    HW4_AVAILABLE = False

# Import HW3 functions from its app.py (without running its server)
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("hw3mod", os.path.join(HW3_DIR, "app.py"))
    hw3mod = importlib.util.module_from_spec(spec) if spec and spec.loader else None
    if spec and spec.loader and hw3mod:
        spec.loader.exec_module(hw3mod)  # type: ignore
    HW3_AVAILABLE = hw3mod is not None
except Exception:
    hw3mod = None
    HW3_AVAILABLE = False

app = Flask(__name__, static_folder="static", template_folder="templates")


# -----------------------------------------------------------------------------
# Top-level routes
# -----------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template(
        "index.html",
        hw1_available=True,
        hw2_available=True,
        hw3_available=HW3_AVAILABLE,
        hw4_available=HW4_AVAILABLE,
        hw7_available=True,
    )


@app.route("/hw1")
def hw1_page():
    return render_template("hw1.html")


@app.route("/hw2")
def hw2_page():
    return render_template("hw2.html")


@app.route("/hw3")
def hw3_page():
    return render_template("hw3.html")


@app.route("/hw4")
def hw4_page():
    return render_template("hw4.html")


@app.route("/hw7")
def hw7_page():
    return render_template("hw7.html")

@app.route("/hw5")
def hw5_page():
    return render_template("hw5.html")


# -----------------------------------------------------------------------------
# HW2 Part-3 endpoints (Template matching + blur)
# -----------------------------------------------------------------------------
HW2_UPLOAD_DIR = os.path.join(app.static_folder, "hw2", "uploads")
HW2_OUTPUT_DIR = os.path.join(app.static_folder, "hw2", "outputs")
HW2_TEMPLATE_DIR = os.path.join(HW2_P3_DIR, "templates_db")
os.makedirs(HW2_UPLOAD_DIR, exist_ok=True)
os.makedirs(HW2_OUTPUT_DIR, exist_ok=True)


def hw2_load_templates() -> List[Tuple[str, np.ndarray]]:
    templates: List[Tuple[str, np.ndarray]] = []
    if not os.path.isdir(HW2_TEMPLATE_DIR):
        return templates
    for fname in sorted(os.listdir(HW2_TEMPLATE_DIR)):
        path = os.path.join(HW2_TEMPLATE_DIR, fname)
        img = cv2.imread(path)
        if img is not None:
            templates.append((fname, img))
    return templates


HW2_TEMPLATES = hw2_load_templates()


def hw2_non_max_suppression(boxes, scores, threshold=0.3):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]
    return keep


def hw2_process_template_matching(image_path: str) -> str:
    scene = cv2.imread(image_path)
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    detected_boxes = []
    scores = []
    labels = []
    MATCH_THRESHOLD = 0.84
    for filename, temp in HW2_TEMPLATES:
        temp_gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        for scale in np.linspace(0.5, 1.5, 15):
            resized = cv2.resize(temp_gray, None, fx=scale, fy=scale)
            tH, tW = resized.shape[:2]
            if tH > scene_gray.shape[0] or tW > scene_gray.shape[1]:
                continue
            result = cv2.matchTemplate(scene_gray, resized, cv2.TM_CCOEFF_NORMED)
            yloc, xloc = np.where(result >= MATCH_THRESHOLD)
            for (x, y) in zip(xloc, yloc):
                detected_boxes.append([x, y, x + tW, y + tH])
                scores.append(result[y, x])
                labels.append(filename)
    keep = hw2_non_max_suppression(detected_boxes, scores)
    output = scene.copy()
    for i in keep:
        x1, y1, x2, y2 = detected_boxes[i]
        label = labels[i]
        roi = output[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(roi, (45, 45), 0)
        output[y1:y2, x1:x2] = blurred
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(output, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    filename = f"output_{uuid.uuid4().hex}.png"
    output_path = os.path.join(HW2_OUTPUT_DIR, filename)
    cv2.imwrite(output_path, output)
    # Return URL path for the client
    return f"/hw2/output/{filename}"


@app.post("/hw2/upload")
def hw2_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    filename = secure_filename(file.filename)
    if not filename:
        filename = f"upload_{uuid.uuid4().hex}.png"
    upload_path = os.path.join(HW2_UPLOAD_DIR, filename)
    file.save(upload_path)
    output_url = hw2_process_template_matching(upload_path)
    return jsonify({"output_url": output_url})


@app.get("/hw2/output/<filename>")
def hw2_output(filename: str):
    return send_from_directory(HW2_OUTPUT_DIR, filename)


# -----------------------------------------------------------------------------
# HW3 endpoints (wrapping imported processing functions)
# -----------------------------------------------------------------------------
def _hw3_ok() -> bool:
    return hw3mod is not None


@app.post("/hw3/run_part1")
def hw3_run_part1():
    if not _hw3_ok():
        return jsonify(success=False, message="HW3 module not available.")
    files = request.files.getlist("images")
    if not files:
        return jsonify(success=False, message="No images uploaded.")
    dataset_dir = hw3mod.save_uploaded_files(files, os.path.join(hw3mod.UPLOAD_ROOT, "part1"))  # type: ignore
    count = hw3mod.process_part1(dataset_dir, hw3mod.PART1_OUT_DIR)  # type: ignore
    return jsonify(success=True, message=f"Part 1 completed for {count} images.",
                   output_dir=os.path.relpath(hw3mod.PART1_OUT_DIR, hw3mod.BASE_DIR))  # type: ignore


@app.post("/hw3/run_part2")
def hw3_run_part2():
    if not _hw3_ok():
        return jsonify(success=False, message="HW3 module not available.")
    files = request.files.getlist("images")
    if not files:
        return jsonify(success=False, message="No images uploaded.")
    dataset_dir = hw3mod.save_uploaded_files(files, os.path.join(hw3mod.UPLOAD_ROOT, "part2"))  # type: ignore
    count = hw3mod.process_part2(dataset_dir, hw3mod.PART2_OUT_DIR)  # type: ignore
    return jsonify(success=True, message=f"Part 2 completed for {count} images.",
                   output_dir=os.path.relpath(hw3mod.PART2_OUT_DIR, hw3mod.BASE_DIR))  # type: ignore


@app.post("/hw3/run_part3")
def hw3_run_part3():
    if not _hw3_ok():
        return jsonify(success=False, message="HW3 module not available.")
    files = request.files.getlist("images")
    if not files:
        return jsonify(success=False, message="No images uploaded.")
    dataset_dir = hw3mod.save_uploaded_files(files, os.path.join(hw3mod.UPLOAD_ROOT, "part3"))  # type: ignore
    count = hw3mod.process_part3(dataset_dir, hw3mod.PART3_OUT_DIR)  # type: ignore
    return jsonify(success=True, message=f"Part 3 completed for {count} images.",
                   output_dir=os.path.relpath(hw3mod.PART3_OUT_DIR, hw3mod.BASE_DIR))  # type: ignore


@app.post("/hw3/run_part4")
def hw3_run_part4():
    if not _hw3_ok():
        return jsonify(success=False, message="HW3 module not available.")
    files = request.files.getlist("images")
    if not files:
        return jsonify(success=False, message="No images uploaded.")
    dataset_dir = hw3mod.save_uploaded_files(files, os.path.join(hw3mod.UPLOAD_ROOT, "part4"))  # type: ignore
    count = hw3mod.process_part4(dataset_dir, hw3mod.PART4_OUT_DIR)  # type: ignore
    return jsonify(success=True, message=f"Part 4 completed for {count} images.",
                   output_dir=os.path.relpath(hw3mod.PART4_OUT_DIR, hw3mod.BASE_DIR))  # type: ignore


@app.post("/hw3/run_part5")
def hw3_run_part5():
    if not _hw3_ok():
        return jsonify(success=False, message="HW3 module not available.")
    image_files = request.files.getlist("images")
    mask_files = request.files.getlist("masks")
    if not image_files:
        return jsonify(success=False, message="No dataset images uploaded for Part 5.")
    dataset_dir = hw3mod.save_uploaded_files(image_files, os.path.join(hw3mod.UPLOAD_ROOT, "part5_dataset"))  # type: ignore
    if mask_files and mask_files[0].filename:
        sam2_dir = hw3mod.save_uploaded_files(mask_files, os.path.join(hw3mod.UPLOAD_ROOT, "part5_sam2_masks"))  # type: ignore
    else:
        sam2_dir = hw3mod.PART5_SAM2_MASK_DIR  # type: ignore
        ok, msg = hw3mod.process_sam2_for_dir(dataset_dir, sam2_dir)  # type: ignore
        if not ok:
            return jsonify(success=False, message=msg)
    count, metrics = hw3mod.process_part5(dataset_dir, sam2_dir, hw3mod.PART5_OUT_DIR)  # type: ignore
    return jsonify(success=True,
                   message=f"Part 5 completed for {count} images.",
                   output_dir=os.path.relpath(hw3mod.PART5_OUT_DIR, hw3mod.BASE_DIR),  # type: ignore
                   metrics=metrics)


# -----------------------------------------------------------------------------
# HW4 endpoints (using imported pure functions)
# -----------------------------------------------------------------------------
HW4_UPLOAD_DIR = os.path.join(app.static_folder, "hw4", "uploads")
HW4_OUTPUT_DIR = os.path.join(app.static_folder, "hw4", "outputs")
os.makedirs(HW4_UPLOAD_DIR, exist_ok=True)
os.makedirs(HW4_OUTPUT_DIR, exist_ok=True)


@app.post("/hw4/process_part1")
def hw4_process_part1():
    if not HW4_AVAILABLE:
        return "HW4 modules not available", 500
    images = request.files.getlist("pano_images")
    saved_paths = []
    for f in images:
        filename = secure_filename(f.filename) or f"pano_{uuid.uuid4().hex}.png"
        path = os.path.join(HW4_UPLOAD_DIR, filename)
        f.save(path)
        saved_paths.append(path)
    output_file = run_panorama_stitching(saved_paths, HW4_OUTPUT_DIR)
    # Return a URL path usable directly by the frontend
    rel = os.path.relpath(output_file, start=os.path.join(app.static_folder, "hw4"))
    return f"/static/hw4/{rel}"


@app.post("/hw4/process_part2")
def hw4_process_part2():
    if not HW4_AVAILABLE:
        return "HW4 modules not available", 500
    img1 = request.files.get("sift_img1")
    img2 = request.files.get("sift_img2")
    if not img1 or not img2:
        return "Two images required", 400
    p1 = os.path.join(HW4_UPLOAD_DIR, secure_filename(img1.filename) or f"img1_{uuid.uuid4().hex}.png")
    p2 = os.path.join(HW4_UPLOAD_DIR, secure_filename(img2.filename) or f"img2_{uuid.uuid4().hex}.png")
    img1.save(p1)
    img2.save(p2)
    output_ours, output_cv = run_sift_comparison(p1, p2, HW4_OUTPUT_DIR)
    rel1 = os.path.relpath(output_ours, start=os.path.join(app.static_folder, "hw4"))
    rel2 = os.path.relpath(output_cv, start=os.path.join(app.static_folder, "hw4"))
    return f"/static/hw4/{rel1}||/static/hw4/{rel2}"


# -----------------------------------------------------------------------------
# HW5 endpoints (launch local OpenCV demos; generate NPZ from bbox)
# -----------------------------------------------------------------------------
HW5_NPZ_DIR = os.path.join(app.static_folder, "hw5", "npz")
os.makedirs(HW5_NPZ_DIR, exist_ok=True)

@app.post("/hw5/make_npz")
def hw5_make_npz():
    """
    Accepts JSON: { "x": int, "y": int, "w": int, "h": int }
    Saves an NPZ with init_bbox and returns its URL path and server path.
    """
    data = request.get_json(silent=True) or {}
    try:
        x = int(data.get("x", 0))
        y = int(data.get("y", 0))
        w = int(data.get("w", 0))
        h = int(data.get("h", 0))
        if w <= 0 or h <= 0:
            return jsonify(success=False, message="Invalid bbox."), 400
    except Exception:
        return jsonify(success=False, message="Invalid bbox payload."), 400

    filename = f"bbox_{uuid.uuid4().hex}.npz"
    npz_path = os.path.join(HW5_NPZ_DIR, filename)
    bbox = np.array([x, y, w, h], dtype=np.float32)
    np.savez(npz_path, init_bbox=bbox)
    return jsonify(success=True,
                   npz_url=f"/static/hw5/npz/{filename}",
                   npz_path=npz_path)

@app.post("/hw5/start")
def hw5_start():
    """
    Launches HW5-3.py in a background subprocess.
    JSON: { "mode": "aruco" | "markerless" | "sam2", "npz_path": optional }
    Returns { success, pid }.
    """
    payload = request.get_json(silent=True) or {}
    mode = str(payload.get("mode", "")).strip()
    if mode not in ("aruco", "markerless", "sam2"):
        return jsonify(success=False, message="Invalid mode."), 400

    args = [sys.executable, os.path.join(HW5_DIR, "HW5-3.py"), "--mode", mode]
    if mode == "sam2":
        npz_path = payload.get("npz_path")
        if not npz_path or not os.path.exists(npz_path):
            return jsonify(success=False, message="npz_path required for sam2 mode."), 400
        args.extend(["--npz", npz_path])

    # Launch non-blocking; OpenCV windows will appear on the local machine.
    proc = subprocess.Popen(args, cwd=HW5_DIR)
    return jsonify(success=True, pid=proc.pid)

if __name__ == "__main__":
    app.run(debug=True)

