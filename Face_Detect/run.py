from flask import Flask, Response, render_template, request
from detector import Detector

app = Flask(__name__)

det = Detector(model_path="my_model.pt", source="1", thresh=0.5, cooling=3.0)

@app.route('/')
def index():
    return render_template('index.html', muted=det.muted)

@app.route('/start')
def start():
    det.start()
    return "started"

@app.route('/stop')
def stop():
    det.stop()
    return "stopped"

@app.route('/mute')
def mute():
    det.toggle_mute()
    return "muted" if det.muted else "unmuted"

@app.route('/video_feed')
def video_feed():
    return Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

def stream():
    for frame in det.frame_generator():
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
