function uploadImage() {
    let fileInput = document.getElementById("imageInput");
    if (fileInput.files.length === 0) {
        alert("Please select an image.");
        return;
    }
    let file = fileInput.files[0];
    document.getElementById("originalImage").src = URL.createObjectURL(file);
    let formData = new FormData();
    formData.append("file", file);
    fetch("/hw2/upload", {
        method: "POST",
        body: formData,
    })
    .then(res => res.json())
    .then(data => {
        if (data.output_url) {
            document.getElementById("outputImage").src = data.output_url;
        } else if (data.error) {
            alert(data.error);
        }
    })
    .catch(err => console.error(err));
}


