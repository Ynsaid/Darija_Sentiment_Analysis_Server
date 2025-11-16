from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)

# Paths to your model and tokenizer
H5_FILE_PATH = r'C:\Users\Younes\Desktop\M2\Nlp\Project\sentiment_cnn_model.h5'
PKL_FILE_PATH = r'C:\Users\Younes\Desktop\M2\Nlp\Project\tokenizer.pkl'

# Load model and tokenizer once at startup
model = load_model(H5_FILE_PATH)
with open(PKL_FILE_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

MAX_LEN = 100  # Change this to your model's input length

labels = ['negative', 'neutral', 'positive']  # Order must match your model output

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    try:
        data = request.get_json()
        if 'text' not in data:
            return jsonify({"error": "الرجاء إرسال 'text' في جسم الطلب"}), 400

        text_to_analyze = data['text']

        # Preprocess text
        seq = tokenizer.texts_to_sequences([text_to_analyze])
        padded = pad_sequences(seq, maxlen=MAX_LEN)

        # Predict
        pred_probs = model.predict(padded)[0]  # Probabilities for each class
        predicted_index = pred_probs.argmax()
        predicted_label = labels[predicted_index]

        # Build confidence dictionary
        confidences_dict = {labels[i]: float(pred_probs[i]) for i in range(len(labels))}

        return jsonify({
            "predicted_label": predicted_label,
            "confidence": confidences_dict,
            "text": text_to_analyze
        })

    except Exception as e:
        print(f"!!! خطأ في /predict: {e}")
        return jsonify({
            "status": "error",
            "message": f"حدث خطأ في الخادم: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
