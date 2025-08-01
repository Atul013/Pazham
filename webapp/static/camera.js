const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const snapButton = document.getElementById('snap');
const cameraForm = document.getElementById('cameraForm');
const cameraImageInput = document.getElementById('cameraImage');

// Access the device camera and stream to video element
navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => {
    video.srcObject = stream;
})
.catch(err => {
    console.error("Error accessing camera: ", err);
});

snapButton.addEventListener('click', () => {
    // Draw the video frame to the canvas
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    // Convert canvas to blob and create a file to submit
    canvas.toBlob(blob => {
        const file = new File([blob], 'captured_image.png', { type: 'image/png' });
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        cameraImageInput.files = dataTransfer.files;
        // Submit the form
        cameraForm.submit();
    }, 'image/png');
});
