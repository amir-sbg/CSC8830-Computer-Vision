import os
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from part1_stitching import run_panorama_stitching
from part2_sift import run_sift_comparison

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER


@app.route("/")
def index():
    return render_template("index.html")


##############################################
# PART 1: PANORAMA STITCHING
##############################################
@app.route("/process_part1", methods=["POST"])
def process_part1():
    images = request.files.getlist("pano_images")
    saved_paths = []

    for f in images:
        filename = secure_filename(f.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(path)
        saved_paths.append(path)

    output_file = run_panorama_stitching(saved_paths, OUTPUT_FOLDER)

    return output_file.replace("static/", "")


##############################################
# PART 2: SIFT FROM SCRATCH
##############################################
@app.route("/process_part2", methods=["POST"])
def process_part2():
    img1 = request.files["sift_img1"]
    img2 = request.files["sift_img2"]

    p1 = os.path.join(UPLOAD_FOLDER, secure_filename(img1.filename))
    p2 = os.path.join(UPLOAD_FOLDER, secure_filename(img2.filename))
    img1.save(p1)
    img2.save(p2)

    output_ours, output_cv = run_sift_comparison(p1, p2, OUTPUT_FOLDER)

    return f"{output_ours.replace('static/','')}||{output_cv.replace('static/','')}"


##############################################
# Serve output files
##############################################
@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True)
