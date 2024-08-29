from flask import Flask, request, jsonify # type:ignore 
from flask_cors import CORS # type:ignore
from color_assignment import assign_colors
from PIL import Image
import io
import numpy as np
import cv2


from color_palette import U2NET_MODEL, preprocess_image, apply_mask, plot_extended_palette

app = Flask(__name__)
CORS(app)


### end point for generating color palette
######################################################################

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Access the raw image data from the request body
        image_data = request.data
        
        if not image_data:
            return jsonify({'error': 'No image data found in the request'}), 400

        # Convert the raw image data to a PIL Image
        image = Image.open(io.BytesIO(image_data))

        # Process the image as before
        image_size = 256
        input_array = preprocess_image(image, image_size)
        y_pred = U2NET_MODEL.predict(input_array)
        predicted_mask = y_pred[0]
        predicted_mask = cv2.resize(predicted_mask, (image_size, image_size))
        original_image = np.array(image.resize((image_size, image_size)))

        focal_object = apply_mask(original_image, predicted_mask)
        extended_palette = plot_extended_palette(focal_object, n_colors=5, n_new_colors=3)

        return jsonify({
            'color_palette': extended_palette,
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# curl -X POST http://localhost:5150/api/process-image --data-binary "@/home/sherif-hodhod/Desktop/bag_design.jpg" -H "Content-Type: image/jpeg"

    


### end point for color assignment
######################################################################

@app.route('/assign_colors', methods=['POST'])
def assign_colors_endpoint():
    try:
        data = request.json
        if not data or 'layers' not in data or 'palette' not in data:
            return jsonify({"error": "Invalid data"}), 400
        
        layers = data['layers']
        palette = data['palette']
        
        assignment = assign_colors(palette, layers)
        return jsonify(assignment)
    
    except Exception as e:
        print("Error occurred:", str(e))  # Print error message
        return jsonify({"error": "Internal server error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
