
# rest api operasyonlarını gradio üzerinde yapmaya karar verdik


'''
from flask import Flask, request, jsonify
import os

# Add libraries for deep learning video processing (e.g., tensorflow, torch)

# Initialize Flask app
app = Flask(__name__)

# Define allowed video extensions
ALLOWED_EXTENSIONS = {'mp4'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for video summarization
@app.route('/summarize', methods=['POST'])
def summarize():
    # Check for uploaded video file
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if video_file and allowed_file(video_file.filename):
        # Get filename and save video
        filename = secure_filename(video_file.filename)
        video_path = os.path.join('uploads', filename)
        video_file.save(video_path)

        # Perform video summarization using deep learning (replace with your logic)
        summarized_video = process_video(video_path)  # Replace with your processing function

        # Delete uploaded video (optional)
        ##os.remove(video_path)

        # Return summarized video data
        return jsonify({'summarized_video': summarized_video}), 200  # Adjust content type if needed
    else:
        return jsonify({'error': 'Unsupported file format'}), 415

# Function for deep learning video processing (replace with your implementation)
@app.route('/get_output', methods=['GET'])
def process_video(video_path):
    # Your deep learning video summarization logic goes here
    # This function should return the processed/summarized video data
    # (e.g., encoded bytes, path to saved video)
    # ... (your deep learning processing logic) ...
    pass

if __name__ == '__main__':
    app.run(debug=True)
'''