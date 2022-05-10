from flask import Flask,render_template,url_for,request
import pickle
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import contractions

tok_file = open('./tokenizer/tokenizer.pickle', 'rb')
tokenizer = pickle.load(tok_file)
tok_file.close()

model = keras.models.load_model('./model/model_glove_humor.h5')

app = Flask(__name__)
def preprocess_text(msg):
    
    sent_length = 10
    text = contractions.fix(msg[0])
    text = text.lower()
    X = [text]
    X_seq = tokenizer.texts_to_sequences(X)
    X_padded = pad_sequences(X_seq,padding='post',maxlen=sent_length, truncating='post')

    return X_padded

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        X_padded = preprocess_text(data)
        pred = model.predict(X_padded)[0][0]
        label = -1
        if pred >= 0.5:
            label = 1
        else:
            label = 0
        print(pred)
        pred = float(str(pred)[:5])
    return render_template('result.html',label = label, pred = pred, data = data[0])


if __name__ == '__main__':
	app.run(debug=True)