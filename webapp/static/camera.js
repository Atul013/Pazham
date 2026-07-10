const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const snapButton = document.getElementById('snap');
const toggleButton = document.getElementById('toggleCamera');
const cameraStatus = document.getElementById('cameraStatus');
const cameraPlaceholder = document.querySelector('.cprev .ph');
const cameraForm = document.getElementById('cameraForm');
const cameraImageInput = document.getElementById('cameraImage');

let stream = null;

function setStatus(msg, isError = false) {
    if (!cameraStatus) return;
    cameraStatus.textContent = msg || '';
    cameraStatus.classList.toggle('is-error', isError);
}

function setActiveUI(active) {
    snapButton.disabled = !active;
    if (toggleButton) {
        toggleButton.innerHTML = active
            ? '<i class="fas fa-video-slash"></i> Stop Camera'
            : '<i class="fas fa-video"></i> Start Camera';
    }
    if (cameraPlaceholder) cameraPlaceholder.style.display = active ? 'none' : 'flex';
}

async function startCamera() {
    if (stream) return;
    setStatus('Requesting camera access…');
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        setActiveUI(true);
        setStatus('');
    } catch (err) {
        console.error('Error accessing camera: ', err);
        stream = null;
        setActiveUI(false);
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
            setStatus('Camera permission denied. You can still upload a photo instead.', true);
        } else if (err.name === 'NotFoundError') {
            setStatus('No camera found on this device.', true);
        } else {
            setStatus('Could not access the camera. Try uploading an image instead.', true);
        }
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    video.srcObject = null;
    setActiveUI(false);
    setStatus('');
}

if (toggleButton) {
    setActiveUI(false);
    toggleButton.addEventListener('click', () => {
        if (stream) stopCamera();
        else startCamera();
    });
}

snapButton.addEventListener('click', () => {
    if (!stream) return;
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(blob => {
        const file = new File([blob], 'captured_image.png', { type: 'image/png' });
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        cameraImageInput.files = dataTransfer.files;
        cameraForm.submit();
    }, 'image/png');
});

// Stop the camera if the user navigates away, so the browser's
// recording indicator doesn't stay on longer than necessary.
window.addEventListener('pagehide', stopCamera);
