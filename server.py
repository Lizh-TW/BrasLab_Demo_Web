import os
import torch
import IPython
import scipy
import numpy as np
import time

from TTS.utils.synthesizer import Synthesizer
from TTS.utils.manage import ModelManager
from TTS.tts.utils.speakers import SpeakerManager

from flask import Flask, render_template, request, make_response, jsonify
from werkzeug.utils import secure_filename


from src.mkdata import sentence_cleaner

# bulid and load config webserver
UPLOAD_FOLDER = './static/save_spekaer_file'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB


use_cuda = False
manager = ModelManager("/workspace/TTS/TTS/.models.json")


model_root = "./model/vits/"
tts_checkpoint_file = model_root + 'checkpoint_1000000.pth.tar'
tts_config_file = model_root + '/config.json'

# speaker_embedding
encoder_checkpoint = './model/se/best_model.pth.tar'
encoder_config = './model/se/config.json'

synthesizer = Synthesizer(tts_checkpoint=tts_checkpoint_file,
                          tts_config_path=tts_config_file,
                          encoder_checkpoint=encoder_checkpoint,
                          encoder_config=encoder_config,
                          use_cuda=use_cuda)

sample_rate = synthesizer.tts_config.audio["sample_rate"]

@app.route("/TTS",  methods=["POST"])
def tts():
    
    file = request.files['TTS_clone_file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    sentence = request.values['TTS_test']
    sentence = sentence_cleaner(sentence)
    
    tts_wav = synthesizer.tts(sentence, speaker_wav=os.path.join(app.config['UPLOAD_FOLDER'], filename))
    tts_wav_filename = time.ctime().replace(' ', '_') + ".wav"
    
    synthesizer.save_wav(tts_wav, os.path.join(app.config['UPLOAD_FOLDER'], tts_wav_filename))
    
    
    return render_template("home.html", rt=1, audio_path=os.path.join(app.config['UPLOAD_FOLDER'], tts_wav_filename), sentence=sentence)

@app.route("/")
def index():
    return render_template("home.html")

def main():
    app.run(debug=True, host="::", port=6006)
    
if __name__ == "__main__":
    main()