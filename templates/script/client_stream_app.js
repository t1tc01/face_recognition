const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const snap  = document.getElementById('snap')

const constraints = {
    audio: false,
    video: {
        width: {min:1024, ideal: 1280, max: 1920},
        height: {min: 576, ideal: 720, max: 1080}
    }
}

async function stopStreamedVideo(videoElem) {
    const stream = videoElem.srcObject;
    const tracks = stream.getTracks();

    tracks.forEach((track) => {
        track.stop();
    });

    videoElem.srcObject = null;
}

async function startWebcam() {
    try {
        await navigator.mediaDevices.getUserMedia(constraints).then((stream) =>
        {
            video.srcObject = stream; // myVideo is an HTML video element
            //imageCapture = new ImageCapture(stream.getVideoTracks()[0]);
            window.stream = stream

        })
        .catch(error => {
            console.error('Error accessing media devices.', error);
        });
    
       
    } catch (e) {
        console.log(e.toString());
    }
}

//POST 
async function postData(data, ctx) {
    const response = await fetch('http://localhost:8008/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data }),
    });
    const result = await response.json();

    ctx.strokeStyle = "green";
    ctx.lineWidth = 5

    if (result['text_name'] !== '') {
        let x = Math.min(result['x1'], result['x2']);
        let y = Math.min(result['y1'], result['y2']);
        let width = Math.abs(result['x2'] - result['x1']);
        let height = Math.abs(result['y2'] - result['y1']);

        ctx.beginPath()
        ctx.rect(x, y, width, height)
        ctx.stroke();
        
        console.log('drawed')
    }
    
}


var context = canvas.getContext('2d')

snap.addEventListener('click', ()=>{
    //context.drawImage(video, 0, 0, 640, 480);
    clearInterval(draw_and_post)
    stopStreamedVideo(video)
});

startWebcam();

draw_and_post = setInterval(() => {
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataURL = canvas.toDataURL('image/png');
    postData(dataURL, context);
});

window.onbeforeunload = function(event) {
// do stuff here
return "you have unsaved changes. Are you sure you want to navigate away?";
};
