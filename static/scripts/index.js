document.addEventListener("DOMContentLoaded", async function () {
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const context = canvas.getContext("2d");
    const captureButton = document.getElementById("capture-btn");

    async function setupCamera() {
        const stream = await navigator.mediaDevices.getUserMedia({ 'video': true });
        video.srcObject = stream;

        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                resolve(video);
            };
        });
    }

    async function init() {
        const video = await setupCamera();
        video.play();

        captureButton.addEventListener("click", async () => {
            context.drawImage(video, 0, 0, 640, 480);
            const imageData = canvas.toDataURL("image/png");

            // Send imageData to server for face recognition
            const response = await fetch("/recognize", {
                method: "POST",
                body: JSON.stringify({ image: imageData }),
                headers: {
                    "Content-Type": "application/json"
                }
            });

            const data = await response.json();
            const resultDiv = document.getElementById("result");
            console.log("Hello " + data.result);

            // Close the window after 5 seconds (5000 milliseconds)
            setTimeout(function(){
                window.close();
            }, 5000);
        });
    }

    init();
});
