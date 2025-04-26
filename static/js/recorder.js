let recordButton = document.getElementById('recordButton');
let status = document.getElementById('status');
let transcriptionDiv = document.getElementById('transcription');

let mediaRecorder;
let audioChunks = [];

recordButton.addEventListener('click', () => {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        recordButton.textContent = "Start Recording";
        status.textContent = "Stopped recording.";
    } else {
        startRecording();
    }
});

function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();
            recordButton.textContent = "Stop Recording";
            status.textContent = "Recording...";

            audioChunks = [];
            mediaRecorder.addEventListener("dataavailable", event => {
                audioChunks.push(event.data);
            });

            mediaRecorder.addEventListener("stop", () => {
const audioBlob = new Blob(audioChunks, { type: 'audio/wav; codecs=1' });
sendAudioToServer(audioBlob);
            });
        })
        .catch(error => {
            status.textContent = "Error accessing microphone: " + error.message;
        });
}

function sendAudioToServer(audioBlob) {
    status.textContent = "Sending audio to server...";
    const formData = new FormData();
    formData.append('audio_data', audioBlob, 'recording.wav');

    fetch('/api/speech-to-text', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.text) {
            transcriptionDiv.textContent = data.text;
            status.textContent = "Transcription received.";
        } else if (data.error) {
            status.textContent = "Error: " + data.error;
        }
    })
    .catch(error => {
        status.textContent = "Error sending audio: " + error.message;
    });
}
