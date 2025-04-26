import torch
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load pretrained model and processor from Hugging Face
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def speech_to_text(audio_path):
    """
    Convert speech audio file to text using Hugging Face Wav2Vec2 model.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        str: Transcribed text from audio.
    """
    # Read audio file with soundfile
    speech, sample_rate = sf.read(audio_path)
    # Resample if needed
    if sample_rate != 16000:
        raise ValueError("Audio sample rate must be 16kHz")

    # Tokenize input
    input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values

    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits

    # Get predicted ids
    predicted_ids = torch.argmax(logits, dim=-1)

    # Decode ids to text
    transcription = processor.decode(predicted_ids[0])

    return transcription
