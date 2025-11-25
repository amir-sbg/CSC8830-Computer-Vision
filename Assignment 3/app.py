import os
import glob
import math
from datetime import datetime

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

import cv2
import numpy as np

# Try to import SAM2 (for part 5). If not installed, we will skip auto-SAM2.
try:
    from ultralytics import SAM
    HAS_SAM2 = True
except Exception:
    HAS_SAM2 = False

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PART1_DIR = os.path.join(BASE_DIR, "part1")
PART2_DIR = os.path.join(BASE_DIR, "part2")
PART3_DIR = os.path.join(BASE_DIR, "part3")
PART4_DIR = os.path.join(BASE_DIR, "part4")
PART5_DIR = os.path.join(BASE_DIR, "part5")

# output dirs as used in your original scripts
PART1_OUT_DIR = os.path.join(PART1_DIR, "output_task1")
PART2_OUT_DIR = os.path.join(PART2_DIR, "output_part2")
PART3_OUT_DIR = os.path.join(PART3_DIR, "output_part3")
PART4_OUT_DIR = os.path.join(PART4_DIR, "output")
PART5_OUT_DIR = os.path.join(PART5_DIR, "output_aruco_seg_2")
PART5_SAM2_MASK_DIR = os.path.join(PART5_DIR, "sam2_masks")

UPLOAD_ROOT = os.path.join(BASE_DIR, "web_uploads")
os.makedirs(UPLOAD_ROOT, exist_ok=True)

# ---------------------------------------------------------------------
# PART 1 HELPERS  (gradient + LoG)
# ---------------------------------------------------------------------

def normalize_to_uint8(img):
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    img = (img - mn) / (mx - mn)
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def compute_gradient_and_log(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = np.arctan2(gy, gx)

    grad_mag_u8 = normalize_to_uint8(mag)

    ang_deg = np.degrees(ang)
    ang_deg[ang_deg < 0] += 180.0
    grad_ang_u8 = normalize_to_uint8(ang_deg)

    blur = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.0, sigmaY=1.0)
    log = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
    log_u8 = normalize_to_uint8(log)

    return grad_mag_u8, grad_ang_u8, log_u8


def process_part1(dataset_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(dataset_dir, "*.*")))
    if not image_paths:
        return 0

    count = 0
    for path in image_paths:
        fname = os.path.splitext(os.path.basename(path))[0]
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        grad_mag_u8, grad_ang_u8, log_u8 = compute_gradient_and_log(gray)

        cv2.imwrite(os.path.join(out_dir, f"{fname}_grad_mag.png"), grad_mag_u8)
        cv2.imwrite(os.path.join(out_dir, f"{fname}_grad_ang.png"), grad_ang_u8)
        cv2.imwrite(os.path.join(out_dir, f"{fname}_log.png"), log_u8)
        count += 1
    return count


# ---------------------------------------------------------------------
# PART 2 HELPERS  (edge + corner keypoints)
# ---------------------------------------------------------------------

EDGE_BLUR_KSIZE = (5, 5)
EDGE_SOBEL_KSIZE = 3
EDGE_MAG_THRESH_RATIO = 0.50

CORNER_BLUR_KSIZE = (7, 7)
CORNER_BLOCK_SIZE = 5
CORNER_KSIZE = 5
CORNER_K = 0.04
CORNER_THRESH_RATIO = 0.07
CORNER_MAX_KP = 200


def detect_edge_keypoints(gray,
                          blur_ksize=EDGE_BLUR_KSIZE,
                          sobel_ksize=EDGE_SOBEL_KSIZE,
                          mag_thresh_ratio=EDGE_MAG_THRESH_RATIO):
    gray_blur = cv2.GaussianBlur(gray, blur_ksize, 1.2)
    gx = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=sobel_ksize)
    gy = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=sobel_ksize)
    mag = cv2.magnitude(gx, gy)

    mag_max = mag.max()
    if mag_max < 1e-6:
        return []

    thresh = mag_thresh_ratio * mag_max
    strong = (mag > thresh).astype(np.uint8)

    mag_dilated = cv2.dilate(mag, np.ones((3, 3), np.uint8))
    local_max = (mag == mag_dilated).astype(np.uint8)

    edge_mask = strong * local_max
    ys, xs = np.where(edge_mask > 0)
    keypoints = [cv2.KeyPoint(float(x), float(y), 3) for x, y in zip(xs, ys)]

    MAX_EDGES = 500
    if len(keypoints) > MAX_EDGES:
        step = len(keypoints) // MAX_EDGES + 1
        keypoints = keypoints[::step]
    return keypoints


def detect_corner_keypoints(gray,
                            blur_ksize=CORNER_BLUR_KSIZE,
                            block_size=CORNER_BLOCK_SIZE,
                            ksize=CORNER_KSIZE,
                            k=CORNER_K,
                            harris_thresh_ratio=CORNER_THRESH_RATIO,
                            max_kp=CORNER_MAX_KP):
    gray_blur = cv2.GaussianBlur(gray, blur_ksize, 2.0)
    harris = cv2.cornerHarris(gray_blur, block_size, ksize, k)

    h_max = harris.max()
    if h_max < 1e-6:
        return []

    thresh = harris_thresh_ratio * h_max
    strong = (harris > thresh)
    h_dilated = cv2.dilate(harris, np.ones((3, 3), np.uint8))
    local_max = (harris == h_dilated)

    corner_mask = strong & local_max
    ys, xs = np.where(corner_mask)
    keypoints = [cv2.KeyPoint(float(x), float(y), 7) for x, y in zip(xs, ys)]

    if len(keypoints) > max_kp:
        step = len(keypoints) // max_kp + 1
        keypoints = keypoints[::step]
    return keypoints


def process_part2(dataset_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    img_paths = sorted(glob.glob(os.path.join(dataset_dir, "*.*")))
    if not img_paths:
        return 0

    count = 0
    for path in img_paths:
        fname = os.path.splitext(os.path.basename(path))[0]
        bgr = cv2.imread(path)
        if bgr is None:
            continue

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        edge_kp = detect_edge_keypoints(gray)
        vis_edge = cv2.drawKeypoints(
            bgr, edge_kp, None,
            color=(0, 0, 255),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        cv2.imwrite(os.path.join(out_dir, f"{fname}_edge_kp.png"), vis_edge)

        corner_kp = detect_corner_keypoints(gray)
        vis_corner = cv2.drawKeypoints(
            bgr, corner_kp, None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        cv2.imwrite(os.path.join(out_dir, f"{fname}_corner_kp.png"), vis_corner)
        count += 1
    return count


# ---------------------------------------------------------------------
# PART 3 HELPERS  (brain boundary)
# ---------------------------------------------------------------------

def get_object_boundary(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        return None, None

    main_cnt = max(contours, key=cv2.contourArea)

    obj_mask = np.zeros_like(mask)
    cv2.drawContours(obj_mask, [main_cnt], -1, 255, thickness=-1)

    overlay = bgr.copy()
    cv2.drawContours(overlay, [main_cnt], -1, (0, 0, 255), thickness=2)

    return obj_mask, overlay


def process_part3(dataset_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(dataset_dir, "*.*")))
    if not image_paths:
        return 0

    count = 0
    for path in image_paths:
        fname = os.path.splitext(os.path.basename(path))[0]
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        obj_mask, overlay = get_object_boundary(bgr)
        if obj_mask is None:
            continue
        cv2.imwrite(os.path.join(out_dir, f"{fname}_mask.png"), obj_mask)
        cv2.imwrite(os.path.join(out_dir, f"{fname}_boundary.png"), overlay)
        count += 1
    return count


# ---------------------------------------------------------------------
# PART 4 + 5 SHARED (ArUco helpers)
# ---------------------------------------------------------------------

CANDIDATE_DICT_NAMES = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9",
    "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11",
]


def detect_aruco_any(gray):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("Your OpenCV build has no cv2.aruco module.")

    best_corners, best_ids, best_dict_name = [], None, None
    max_markers = 0

    for name in CANDIDATE_DICT_NAMES:
        if not hasattr(cv2.aruco, name):
            continue

        dict_id = getattr(cv2.aruco, name)
        dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

        if hasattr(cv2.aruco, "DetectorParameters") and hasattr(cv2.aruco, "ArucoDetector"):
            params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(dictionary, params)
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            params = cv2.aruco.DetectorParameters_create()
            corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)

        num = 0 if ids is None else len(ids)
        if num > max_markers:
            max_markers = num
            best_corners, best_ids, best_dict_name = corners, ids, name

    return best_corners, best_ids, best_dict_name


def compute_marker_sizes(corners, ids):
    sizes = []
    if corners is None or len(corners) == 0:
        return sizes

    for i, c in enumerate(corners):
        pts = c.reshape(-1, 2).astype(np.float32)
        side_lengths = []
        for j in range(4):
            p1 = pts[j]
            p2 = pts[(j + 1) % 4]
            side_lengths.append(np.linalg.norm(p1 - p2))
        avg_side = float(np.mean(side_lengths))

        area = float(cv2.contourArea(pts))
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        marker_id = int(ids[i][0]) if ids is not None else -1

        sizes.append({
            "id": marker_id,
            "side": avg_side,
            "area": area,
            "center": (cx, cy),
        })
    return sizes


def segment_object_with_aruco(bgr):
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    corners, ids, dict_name = detect_aruco_any(gray_blur)
    if ids is None or len(corners) == 0:
        return None, None, None, None, None, None

    dict_id = getattr(cv2.aruco, dict_name)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    if hasattr(cv2.aruco, "DetectorParameters") and hasattr(cv2.aruco, "ArucoDetector"):
        markers_vis = cv2.aruco.drawDetectedMarkers(bgr.copy(), corners, ids)
    else:
        markers_vis = cv2.aruco.drawDetectedMarkers(bgr.copy(), corners, ids)

    pts_list = [c.reshape(-1, 2) for c in corners]
    pts_all = np.vstack(pts_list).astype(np.float32)
    hull = cv2.convexHull(pts_all)

    obj_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(obj_mask, hull.astype(np.int32), 255)

    overlay = bgr.copy()
    cv2.drawContours(overlay, [hull.astype(np.int32)], -1, (0, 0, 255), 3)

    marker_sizes = compute_marker_sizes(corners, ids)
    return markers_vis, obj_mask, overlay, marker_sizes, ids, dict_name


# ---------------------------------------------------------------------
# PART 4 PROCESSOR
# ---------------------------------------------------------------------

def process_part4(dataset_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    img_paths = sorted(glob.glob(os.path.join(dataset_dir, "*.*")))
    if not img_paths:
        return 0

    count = 0
    for path in img_paths:
        fname = os.path.splitext(os.path.basename(path))[0]
        bgr = cv2.imread(path)
        if bgr is None:
            continue

        (markers_vis,
         mask,
         overlay,
         marker_sizes,
         ids,
         dict_name) = segment_object_with_aruco(bgr)

        if mask is None:
            continue

        overlay_with_size = overlay.copy()
        for m in marker_sizes:
            cx, cy = m["center"]
            text = f"{m['side']:.1f}px"
            cv2.putText(
                overlay_with_size,
                text,
                (int(cx) - 20, int(cy) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
                cv2.LINE_AA
            )

        cv2.imwrite(os.path.join(out_dir, f"{fname}_markers.png"), markers_vis)
        cv2.imwrite(os.path.join(out_dir, f"{fname}_mask.png"), mask)
        cv2.imwrite(os.path.join(out_dir, f"{fname}_boundary.png"), overlay)
        cv2.imwrite(os.path.join(out_dir, f"{fname}_size.png"), overlay_with_size)
        count += 1

    return count


# ---------------------------------------------------------------------
# PART 5 HELPERS  (SAM2 + comparison)
# ---------------------------------------------------------------------

def build_sam2_model():
    if not HAS_SAM2:
        raise RuntimeError("ultralytics.SAM is not installed.")
    model = SAM("sam2_b.pt")
    return model


def run_sam2_on_image(model, bgr_img):
    img = bgr_img.copy()
    results = model.predict(source=img, retina_masks=True, verbose=False)

    if not results:
        h, w = img.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)

    r = results[0]
    if r.masks is None or r.masks.data is None or len(r.masks.data) == 0:
        h, w = img.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)

    masks = r.masks.data.cpu().numpy()
    areas = masks.sum(axis=(1, 2))
    best_idx = int(np.argmax(areas))
    best_mask = masks[best_idx]

    binary = (best_mask > 0.5).astype(np.uint8) * 255
    return binary


def compute_iou_and_dice(mask_a, mask_b):
    a = mask_a > 0
    b = mask_b > 0

    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    a_sum = a.sum()
    b_sum = b.sum()

    iou = 0.0 if union == 0 else inter / float(union)
    dice = 0.0 if (a_sum + b_sum) == 0 else 2.0 * inter / float(a_sum + b_sum)
    return iou, dice


def make_comparison_overlay(bgr, mask_aruco, mask_sam2):
    h, w = bgr.shape[:2]
    mask_sam2_resized = cv2.resize(mask_sam2, (w, h), interpolation=cv2.INTER_NEAREST)

    contours_aruco, _ = cv2.findContours(
        (mask_aruco > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours_sam2, _ = cv2.findContours(
        (mask_sam2_resized > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    overlay = bgr.copy()
    cv2.drawContours(overlay, contours_aruco, -1, (0, 0, 255), 2)
    cv2.drawContours(overlay, contours_sam2, -1, (0, 255, 0), 2)
    return overlay, mask_sam2_resized


def process_sam2_for_dir(dataset_dir, mask_dir):
    os.makedirs(mask_dir, exist_ok=True)
    if not HAS_SAM2:
        return False, "SAM2 model not available (ultralytics not installed)."

    model = build_sam2_model()
    img_paths = sorted(glob.glob(os.path.join(dataset_dir, "*.*")))
    for path in img_paths:
        fname = os.path.splitext(os.path.basename(path))[0]
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        mask = run_sam2_on_image(model, bgr)
        out_path = os.path.join(mask_dir, fname + ".png")
        cv2.imwrite(out_path, mask)
    return True, "SAM2 masks generated."


def load_sam2_mask_for_image(mask_dir, img_path):
    base = os.path.splitext(os.path.basename(img_path))[0]
    candidates = []
    for ext in ("png", "jpg", "jpeg"):
        candidates.append(os.path.join(mask_dir, f"{base}_mask.{ext}"))
    for ext in ("png", "jpg", "jpeg"):
        candidates.append(os.path.join(mask_dir, f"{base}.{ext}"))
    for path in candidates:
        if os.path.exists(path):
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                return mask

    wildcard = os.path.join(mask_dir, base + "*")
    for path in glob.glob(wildcard):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            return mask
    return None


def process_part5(dataset_dir, sam2_mask_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    img_paths = sorted(glob.glob(os.path.join(dataset_dir, "*.*")))
    if not img_paths:
        return 0, []

    all_metrics = []

    for path in img_paths:
        fname = os.path.splitext(os.path.basename(path))[0]
        bgr = cv2.imread(path)
        if bgr is None:
            continue

        (markers_vis,
         mask_aruco,
         overlay_aruco,
         marker_sizes,
         ids,
         dict_name) = segment_object_with_aruco(bgr)

        if mask_aruco is None:
            continue

        overlay_with_size = overlay_aruco.copy()
        for m in marker_sizes:
            cx, cy = m["center"]
            text = f"{m['side']:.1f}px"
            cv2.putText(
                overlay_with_size,
                text,
                (int(cx) - 20, int(cy) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
                cv2.LINE_AA
            )

        cv2.imwrite(os.path.join(out_dir, f"{fname}_markers.png"), markers_vis)
        cv2.imwrite(os.path.join(out_dir, f"{fname}_mask_aruco.png"), mask_aruco)
        cv2.imwrite(os.path.join(out_dir, f"{fname}_boundary_aruco.png"), overlay_aruco)
        cv2.imwrite(os.path.join(out_dir, f"{fname}_size_aruco.png"), overlay_with_size)

        sam2_mask = load_sam2_mask_for_image(sam2_mask_dir, path)
        if sam2_mask is None:
            continue

        comp_overlay, sam2_mask_resized = make_comparison_overlay(
            bgr, mask_aruco, sam2_mask
        )

        iou, dice = compute_iou_and_dice(mask_aruco, sam2_mask_resized)
        cv2.imwrite(os.path.join(out_dir, f"{fname}_mask_sam2.png"), sam2_mask_resized)
        cv2.imwrite(os.path.join(out_dir, f"{fname}_compare_overlay.png"), comp_overlay)

        all_metrics.append((fname, float(iou), float(dice)))

    return len(all_metrics), all_metrics


# ---------------------------------------------------------------------
# FLASK APP
# ---------------------------------------------------------------------

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", has_sam2=HAS_SAM2)


def save_uploaded_files(files, target_root):
    """Save uploaded files to a fresh timestamped subfolder."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_dir = os.path.join(target_root, ts)
    os.makedirs(dest_dir, exist_ok=True)

    for f in files:
        if not f.filename:
            continue
        filename = secure_filename(f.filename)
        f.save(os.path.join(dest_dir, filename))
    return dest_dir


@app.post("/run_part1")
def run_part1_route():
    files = request.files.getlist("images")
    if not files:
        return jsonify(success=False, message="No images uploaded.")
    dataset_dir = save_uploaded_files(files, os.path.join(UPLOAD_ROOT, "part1"))
    count = process_part1(dataset_dir, PART1_OUT_DIR)
    return jsonify(
        success=True,
        message=f"Part 1 completed for {count} images.",
        output_dir=os.path.relpath(PART1_OUT_DIR, BASE_DIR),
    )


@app.post("/run_part2")
def run_part2_route():
    files = request.files.getlist("images")
    if not files:
        return jsonify(success=False, message="No images uploaded.")
    dataset_dir = save_uploaded_files(files, os.path.join(UPLOAD_ROOT, "part2"))
    count = process_part2(dataset_dir, PART2_OUT_DIR)
    return jsonify(
        success=True,
        message=f"Part 2 completed for {count} images.",
        output_dir=os.path.relpath(PART2_OUT_DIR, BASE_DIR),
    )


@app.post("/run_part3")
def run_part3_route():
    files = request.files.getlist("images")
    if not files:
        return jsonify(success=False, message="No images uploaded.")
    dataset_dir = save_uploaded_files(files, os.path.join(UPLOAD_ROOT, "part3"))
    count = process_part3(dataset_dir, PART3_OUT_DIR)
    return jsonify(
        success=True,
        message=f"Part 3 completed for {count} images.",
        output_dir=os.path.relpath(PART3_OUT_DIR, BASE_DIR),
    )


@app.post("/run_part4")
def run_part4_route():
    files = request.files.getlist("images")
    if not files:
        return jsonify(success=False, message="No images uploaded.")
    dataset_dir = save_uploaded_files(files, os.path.join(UPLOAD_ROOT, "part4"))
    count = process_part4(dataset_dir, PART4_OUT_DIR)
    return jsonify(
        success=True,
        message=f"Part 4 completed for {count} images.",
        output_dir=os.path.relpath(PART4_OUT_DIR, BASE_DIR),
    )


@app.post("/run_part5")
def run_part5_route():
    image_files = request.files.getlist("images")
    mask_files = request.files.getlist("masks")

    if not image_files:
        return jsonify(success=False, message="No dataset images uploaded for Part 5.")

    dataset_dir = save_uploaded_files(image_files, os.path.join(UPLOAD_ROOT, "part5_dataset"))

    # if mask files were uploaded, save them; otherwise, try to generate via SAM2
    if mask_files and mask_files[0].filename:
        sam2_dir = save_uploaded_files(mask_files, os.path.join(UPLOAD_ROOT, "part5_sam2_masks"))
    else:
        # generate SAM2 masks directly into the official mask dir
        sam2_dir = PART5_SAM2_MASK_DIR
        ok, msg = process_sam2_for_dir(dataset_dir, sam2_dir)
        if not ok:
            return jsonify(success=False, message=msg)

    count, metrics = process_part5(dataset_dir, sam2_dir, PART5_OUT_DIR)

    return jsonify(
        success=True,
        message=f"Part 5 completed for {count} images. (See console for IoU/Dice if needed.)",
        output_dir=os.path.relpath(PART5_OUT_DIR, BASE_DIR),
        metrics=metrics,
    )


if __name__ == "__main__":
    app.run(debug=True)
