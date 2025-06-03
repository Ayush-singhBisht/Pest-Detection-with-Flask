# from flask import Flask, render_template, Response, request, session, redirect, url_for
# from backend.detection import gen_frames
# from backend.upload import predict_uploaded_image
# from backend.upload_vedio import gen_video_frames
# from backend.logger import get_logs
# import os
# from werkzeug.utils import secure_filename

# UPLOAD_FOLDER = 'static/uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# app = Flask(__name__)
# app.secret_key = 'your_key'


# @app.route('/')
# @app.route('/live')
# def live():
#     return render_template('live.html', active_page='live')


# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route('/upload-video', methods=['GET', 'POST'])
# def upload_video():
#     if request.method == 'POST':
#         if 'video' not in request.files:
#             return "No video part in the request", 400

#         file = request.files['video']

#         if file.filename == '':
#             return "No video selected", 400

#         filename = secure_filename(file.filename)
#         file_path = os.path.join(UPLOAD_FOLDER, filename)
#         file.save(file_path)

#         session['uploaded_video_path'] = file_path

#         return redirect(url_for('video_feed_uploaded'))

#     return render_template('upload_video.html', active_page='upload_video')


# @app.route('/video-feed-uploaded')
# def video_feed_uploaded():
#     video_path = session.get('uploaded_video_path')
#     if video_path and os.path.exists(video_path):
#         return Response(gen_video_frames(video_path),
#                         mimetype='multipart/x-mixed-replace; boundary=frame')
#     return "Video not found or expired", 404


# @app.route('/logs')
# def logs():
#     logs_data = get_logs()
#     return render_template('logs.html', logs=logs_data, active_page='logs')


# @app.route('/upload-image', methods=['GET', 'POST'])
# def upload_image():
#     label = None
#     image_url = None
#     if request.method == 'POST':
#         if 'image' not in request.files:
#             label = 'No file part'
#         else:
#             file = request.files['image']
#             if file.filename == '':
#                 label = 'No selected file'
#             else:
#                 filename = secure_filename(file.filename)
#                 file_path = os.path.join(UPLOAD_FOLDER, filename)
#                 file.save(file_path)

#                 label = predict_uploaded_image(file)

#                 image_url = f"/{file_path.replace(os.sep, '/')}"
#     return render_template('upload_image.html', prediction=label, image_url=image_url, active_page='upload_image')


# if __name__ == '__main__':
#     app.run(debug=True)










from flask import Flask, render_template, Response, request, session, redirect, url_for
from backend.detection import gen_frames, camera1, camera2, camera3
from backend.upload import predict_uploaded_image
from backend.upload_vedio import gen_video_frames
from backend.logger import get_logs
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'your_key'


@app.route('/')
@app.route('/live')
def live():
    return render_template('live.html', active_page='live')


# === Three Live Feeds ===
@app.route('/video_feed1')
def video_feed1():
    return Response(gen_frames(camera1, cam_id="Camera 1"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed2')
def video_feed2():
    return Response(gen_frames(camera2, cam_id="Camera 2"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed3')
def video_feed3():
    return Response(gen_frames(camera3, cam_id="Camera 3"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# === Upload Video and Process ===
@app.route('/upload-video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'video' not in request.files:
            return "No video part in the request", 400

        file = request.files['video']
        if file.filename == '':
            return "No video selected", 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        session['uploaded_video_path'] = file_path
        return redirect(url_for('video_feed_uploaded'))

    return render_template('upload_video.html', active_page='upload_video')


@app.route('/video-feed-uploaded')
def video_feed_uploaded():
    video_path = session.get('uploaded_video_path')
    if video_path and os.path.exists(video_path):
        return Response(gen_video_frames(video_path),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    return "Video not found or expired", 404


# === Image Upload Prediction ===
@app.route('/upload-image', methods=['GET', 'POST'])
def upload_image():
    label = None
    image_url = None
    if request.method == 'POST':
        if 'image' not in request.files:
            label = 'No file part'
        else:
            file = request.files['image']
            if file.filename == '':
                label = 'No selected file'
            else:
                filename = secure_filename(file.filename)
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(file_path)

                label = predict_uploaded_image(file)
                image_url = f"/{file_path.replace(os.sep, '/')}"

    return render_template('upload_image.html', prediction=label, image_url=image_url, active_page='upload_image')


# === Logs View ===
@app.route('/logs')
def logs():
    logs_data = get_logs()
    return render_template('logs.html', logs=logs_data, active_page='logs')


if __name__ == '__main__':
    app.run(debug=True)
