from flask import Flask, render_template, request, jsonify
import os
import tempfile
from nlp_intent import detect_intent
from speech2text import speech_to_text
from pydub import AudioSegment

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',image="default.jpg")

@app.route('/api/speech-to-text', methods=['POST'])
def api_speech_to_text():
    if 'audio_data' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio_data']

    wav_filename = "recording.wav"
    try:
        # Save the uploaded file temporarily with a generic suffix
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".tmp")
        audio_file.save(temp_input.name)
        temp_input.close()

        import os
        if not os.path.exists(temp_input.name):
            raise FileNotFoundError(f"Temporary file not found: {temp_input.name}")
        print(f"Temporary file saved at: {temp_input.name}")

        # Convert to WAV format and save as recording.wav in current directory
        audio = AudioSegment.from_file(temp_input.name)
        # Resample to 16kHz
        audio = audio.set_frame_rate(16000)
        audio.export(wav_filename, format="wav")
        file_size = os.path.getsize(wav_filename)
        print(f"Saved converted audio file at: {os.path.abspath(wav_filename)} with size: {file_size} bytes")

        # Remove the temporary input file
        os.remove(temp_input.name)
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return jsonify({'error': f"Error processing audio file: {e}"}), 500

    try:
        # Convert speech to text using speech2text.py
        text = speech_to_text(wav_filename)
        intent = detect_intent(text)
        if intent == "on_light":
            image = "light_on.jpg"
        elif intent == "off_light":
            image = "light_off.jpg"
        elif intent == "on_fan":
            image = "fan_on.jpg"
        elif intent == "off_fan":
            image = "fan_off.jpg"
        else:
            image = "default.jpg"
        #return render_template('index.html', image=image)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    print("************************************",text, image)
    return jsonify({'text': text, 'image': image})

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
