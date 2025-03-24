from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
<<<<<<< HEAD
=======
import os
>>>>>>> a25cc0bd9ca14d2b99e70771cc8f46f5490f65f1

app = Flask(__name__)

# Load the TFLite model
<<<<<<< HEAD
interpreter = tf.lite.Interpreter(model_path="mango_leaf_disease_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
=======
try:
    interpreter = tf.lite.Interpreter(model_path="mango_leaf_disease_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    raise Exception(f"Failed to load TFLite model: {str(e)}")
>>>>>>> a25cc0bd9ca14d2b99e70771cc8f46f5490f65f1

# Define disease information
disease_info = {
    'Anthracnose': {
        'cure': 'Apply copper-based fungicide and remove infected leaves.',
        'cause': 'Caused by the fungus Colletotrichum gloeosporioides, often due to wet conditions.'
    },
    'Bacterial Canker': {
        'cure': 'Prune affected branches and use copper oxychloride.',
        'cause': 'Caused by Xanthomonas campestris pv. mangiferaeindicae, spread through wounds or rain splash.'
    },
    'Cutting Weevil': {
        'cure': 'Use insecticidal soap and remove affected parts.',
        'cause': 'Caused by the insect Sternochetus mangiferae, which lays eggs in leaf tissue.'
    },
    'Die Back': {
        'cure': 'Improve drainage and apply fungicide.',
        'cause': 'Caused by fungi like Lasiodiplodia theobromae, often due to poor soil drainage.'
    },
    'Gall Midge': {
        'cure': 'Use systemic insecticides and monitor regularly.',
        'cause': 'Caused by the insect Procontarinia matteiana, which attacks young leaves.'
    },
    'Healthy': {
        'cure': 'No treatment needed.',
        'cause': 'N/A'
    },
    'Powdery Mildew': {
        'cure': 'Apply sulfur dust or neem oil.',
        'cause': 'Caused by the fungus Oidium mangiferae, often in dry, warm conditions.'
    }
}

# Define class labels
class_labels = [
    'Anthracnose',
    'Bacterial Canker',
    'Cutting Weevil',
    'Die Back',
    'Gall Midge',
    'Healthy',
    'Powdery Mildew'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
<<<<<<< HEAD
=======
        # Parse the JSON request
>>>>>>> a25cc0bd9ca14d2b99e70771cc8f46f5490f65f1
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

<<<<<<< HEAD
        # Decode base64 image
        img_data = base64.b64decode(data['image'])
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        img = img.resize((256, 256), Image.NEAREST)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class_idx = np.argmax(output_data[0])
        disease = class_labels[predicted_class_idx]
        info = disease_info.get(disease, {'cure': 'No cure information available', 'cause': 'Unknown cause'})
=======
        # Decode the base64 image
        try:
            img_data = base64.b64decode(data['image'])
        except Exception as e:
            return jsonify({'error': f'Failed to decode base64 string: {str(e)}'}), 400

        # Open and process the image
        try:
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
        except Exception as e:
            return jsonify({'error': f'Failed to open image: {str(e)}'}), 400

        # Resize the image
        try:
            img = img.resize((256, 256), Image.Resampling.NEAREST)  # Updated for Pillow 10.0.0+
        except Exception as e:
            return jsonify({'error': f'Failed to resize image: {str(e)}'}), 400

        # Convert image to numpy array
        try:
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        except Exception as e:
            return jsonify({'error': f'Failed to convert image to numpy array: {str(e)}'}), 400

        # Run inference
        try:
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_class_idx = np.argmax(output_data[0])
            disease = class_labels[predicted_class_idx]
            info = disease_info.get(disease, {'cure': 'No cure information available', 'cause': 'Unknown cause'})
        except Exception as e:
            return jsonify({'error': f'Failed during model inference: {str(e)}'}), 500
>>>>>>> a25cc0bd9ca14d2b99e70771cc8f46f5490f65f1

        return jsonify({
            'disease': disease,
            'cure': info['cure'],
            'cause': info['cause']
        })
    except Exception as e:
<<<<<<< HEAD
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
=======
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use PORT env variable, default to 5000
    app.run(host='0.0.0.0', port=port, debug=False)  # Debug=False for production
>>>>>>> a25cc0bd9ca14d2b99e70771cc8f46f5490f65f1
