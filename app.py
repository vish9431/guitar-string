from flask import Flask, request, render_template
import librosa
import numpy as np
import pickle

MAX_NUM_FRAMES = 173
SAMPLE_RATE = 44100
DURATION = 2
NUM_MFCCS = 20


class MyFlask(Flask):
    def run(self, host=None, port=None, debug=None, load_dotenv=None, **kwargs):
        if not self.debug or os.getenv('WERKZEUG_RUN_PATH') == 'true':
            with self.app_context():
                global model 
                with open('abc.pkl', 'wb') as file:
                    model = pickle.load(file)
                
        super(MyFlask, self).run(host=host, port=port,
                                 debug=debug, load_dotenv=load_dotenv, **kwargs)


app = MyFlask(__name__)
model = None

defined_class = ['A', 'B', 'D', 'Ehi', 'Elo', 'G']


def convert(file):
    with open(file, 'rb') as f:
        with open('raw.wav', 'wb') as fd:
            fd.write(f.read())

def process(audio_path):
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
    mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=NUM_MFCCS)
    mfccs = mfccs.T  
                
    if mfccs.shape[0] < MAX_NUM_FRAMES:
        mfccs = np.pad(mfccs, ((0, MAX_NUM_FRAMES - mfccs.shape[0]), (0, 0)), 'constant')
    elif mfccs.shape[0] > MAX_NUM_FRAMES:
        mfccs = mfccs[:MAX_NUM_FRAMES, :]
    
    return np.array([mfccs[:, :, np.newaxis]])


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        raw_file = request.files['audioPart']
        raw_file.save('raw.mp3')
        
        convert('raw.mp3')
        data = process('raw.wav')
        prediction = model.predict(data)
        prediction = defined_class[np.argmax(prediction)]
        
        return prediction
    else:
        return render_template('main.html')


if __name__ == '__main__':
    app.run('0.0.0.0')
