let K = [
  [1434.41, 0.00, 949.77],
  [0.00, 1430.68, 541.41],
  [0.00, 0.00, 1.00]
];

let K_inv = math.inv(K);
let org_w, org_h;

let picture = document.getElementById("picture");
let pictureCoords = picture.getBoundingClientRect();
let ruler = document.getElementById("ruler");
let aimer = document.getElementById("aimer");
let rulerCtx = ruler.getContext("2d");
let aimerCtx = aimer.getContext("2d");

function upload(e) {
  let file = e.target.files[0];
  let org_image = new Image();
  org_image.src = URL.createObjectURL(file);
  org_image.onload = () => {
    org_w = org_image.width;
    org_h = org_image.height;
  };
  picture.src = org_image.src;
  picture.onload = () => {
    ruler.width = picture.width;
    ruler.height = picture.height;
    aimer.width = picture.width;
    aimer.height = picture.height;
    pictureCoords = picture.getBoundingClientRect();
  };
}

function showAim(e) {
  aimerCtx.clearRect(0, 0, aimer.width, aimer.height);
  let { clientX, clientY } = e;
  let x = clientX - pictureCoords.left;
  let y = clientY - pictureCoords.top;

  aimerCtx.beginPath();
  aimerCtx.strokeStyle = "#00ff00";
  aimerCtx.moveTo(0, y);
  aimerCtx.lineTo(aimer.width, y);
  aimerCtx.moveTo(x, 0);
  aimerCtx.lineTo(x, aimer.height);
  aimerCtx.stroke();
}
ruler.onmousemove = showAim;

function point(x, y, color = "#00ff00") {
  rulerCtx.beginPath();
  rulerCtx.arc(x, y, 7, 0, 2 * Math.PI);
  rulerCtx.fillStyle = color;
  rulerCtx.fill();
}

let one, two, zc;
let firstClick = true;

function getCoords(e) {
  let { clientX, clientY } = e;
  let x = clientX - pictureCoords.left;
  let y = clientY - pictureCoords.top;

  let scaled_x = x * org_w / picture.width;
  let scaled_y = y * org_h / picture.height;

  if (firstClick) {
    rulerCtx.clearRect(0, 0, aimer.width, aimer.height);
    one = [[scaled_x], [scaled_y], [1]];
    point(x, y, "#ff0000");
  } else {
    two = [[scaled_x], [scaled_y], [1]];
    point(x, y, "#ffff00");
    zc = parseFloat(prompt("Distance from camera (Zc)?"));
    const dist = computeDistance(one, two);
    alert(`Estimated length: ${dist.toFixed(2)}`);
  }

  firstClick = !firstClick;
}
ruler.onclick = getCoords;

// ---- Corrected computation ----
function computeDistance(one, two) {
  // multiply by inverse intrinsic matrix
  const xyz1 = math.multiply(K_inv, one);
  const xyz2 = math.multiply(K_inv, two);

  // scale each by depth (zc)
  for (let i = 0; i < 3; i++) {
    xyz1[i][0] *= zc;
    xyz2[i][0] *= zc;
  }

  // compute Euclidean distance
  const dx = xyz1[0][0] - xyz2[0][0];
  const dy = xyz1[1][0] - xyz2[1][0];
  const dz = xyz1[2][0] - xyz2[2][0];
  const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
  return d;
}
