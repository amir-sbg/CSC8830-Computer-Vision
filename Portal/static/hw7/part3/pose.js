const videoElement = document.getElementById("inputVideo");
const canvasElement = document.getElementById("outputCanvas");
const canvasCtx = canvasElement.getContext("2d");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const downloadCsvBtn = document.getElementById("downloadCsvBtn");
const clearCsvBtn = document.getElementById("clearCsvBtn");
const statusP = document.getElementById("status");
let camera = null;
let holistic = null;
let frameIndex = 0;
let csvRows = [];
holistic = new Holistic({
  locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
  },
});
holistic.setOptions({
  modelComplexity: 1,
  smoothLandmarks: true,
  enableSegmentation: false,
  refineFaceLandmarks: false,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
});
holistic.onResults(onResults);
function onResults(results) {
  canvasElement.width = results.image.width;
  canvasElement.height = results.image.height;
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
  if (results.poseLandmarks) {
    drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, { lineWidth: 4 });
    drawLandmarks(canvasCtx, results.poseLandmarks, { lineWidth: 2 });
    for (let i = 0; i < results.poseLandmarks.length; i++) {
      const lm = results.poseLandmarks[i];
      csvRows.push({ frame: frameIndex, type: "pose", landmark_id: i, x: lm.x, y: lm.y, z: lm.z, visibility: lm.visibility });
    }
  }
  if (results.leftHandLandmarks) {
    drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS, { lineWidth: 3 });
    drawLandmarks(canvasCtx, results.leftHandLandmarks, { lineWidth: 2 });
    for (let i = 0; i < results.leftHandLandmarks.length; i++) {
      const lm = results.leftHandLandmarks[i];
      csvRows.push({ frame: frameIndex, type: "left_hand", landmark_id: i, x: lm.x, y: lm.y, z: lm.z, visibility: lm.visibility });
    }
  }
  if (results.rightHandLandmarks) {
    drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS, { lineWidth: 3 });
    drawLandmarks(canvasCtx, results.rightHandLandmarks, { lineWidth: 2 });
    for (let i = 0; i < results.rightHandLandmarks.length; i++) {
      const lm = results.rightHandLandmarks[i];
      csvRows.push({ frame: frameIndex, type: "right_hand", landmark_id: i, x: lm.x, y: lm.y, z: lm.z, visibility: lm.visibility });
    }
  }
  canvasCtx.restore();
  frameIndex += 1;
}
function startCamera() {
  if (camera) return;
  camera = new Camera(videoElement, {
    onFrame: async () => {
      await holistic.send({ image: videoElement });
    },
    width: 640,
    height: 480,
  });
  camera.start();
  statusP.textContent = "Webcam started. Move your body and hands in front of the camera.";
}
function stopCamera() {
  if (!camera) return;
  camera.stop();
  camera = null;
  statusP.textContent = "Webcam stopped.";
}
function downloadCSV() {
  if (csvRows.length === 0) {
    alert("No data recorded yet.");
    return;
  }
  const header = "frame,type,landmark_id,x,y,z,visibility";
  const lines = [header];
  for (const row of csvRows) {
    const line = [row.frame, row.type, row.landmark_id, row.x, row.y, row.z !== undefined ? row.z : "", row.visibility !== undefined ? row.visibility : ""].join(",");
    lines.push(line);
  }
  const csvContent = lines.join("\n");
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "pose_hand_data.csv";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
function clearCSV() {
  csvRows = [];
  frameIndex = 0;
  statusP.textContent = "CSV buffer cleared. New frames will be recorded.";
}
startBtn.addEventListener("click", () => { startCamera(); });
stopBtn.addEventListener("click", () => { stopCamera(); });
downloadCsvBtn.addEventListener("click", () => { downloadCSV(); });
clearCsvBtn.addEventListener("click", () => { clearCSV(); });


