from flask import Flask, request, jsonify
import pickle
import cv2
import numpy as np
import mediapipe as mp
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)

# Load your model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
labels_dict = {0: 'A', 1: 'B', 2: 'C'}

# Initialize a variable to store the last processed timestamp
last_timestamp = None


def process_frame(frame, timestamp):
    global last_timestamp
    if last_timestamp is not None and timestamp <= last_timestamp:
        raise ValueError(
            f"Received frame with timestamp {timestamp} which is not greater than last timestamp {last_timestamp}")
    last_timestamp = timestamp

    data_aux = []
    x_ = []
    y_ = []

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]
        return predicted_character
    return None


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['frame'].read()
    npimg = np.frombuffer(file, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Extract the timestamp from the request, or use the current time
    timestamp = int(request.form.get('timestamp', 0))

    try:
        prediction = process_frame(frame, timestamp)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
