from flask import Flask, render_template, request, redirect, url_for, flash
import os
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
from model_utils import predict_from_image

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-only-secret-key-change-me')
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.context_processor
def inject_current_year():
    return {'current_year': datetime.now().year}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/team')
def team():
    return render_template('team.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        flash('No image was submitted. Please choose a file or use the camera.')
        return redirect(url_for('index'))
    file = request.files['image']
    if file.filename == '':
        flash('No image was submitted. Please choose a file or use the camera.')
        return redirect(url_for('index'))
    if not allowed_file(file.filename):
        flash('Unsupported file type. Please upload a JPG or PNG image.')
        return redirect(url_for('index'))

    original_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{original_name}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(filepath)
    result = predict_from_image(filepath)
    return render_template('result.html', result=result, filename=unique_name)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename))


@app.errorhandler(413)
def file_too_large(e):
    flash('That image is too large. Please upload a file under 16 MB.')
    return redirect(url_for('index'))


if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug_mode, threaded=True)
