from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize YOLOv8 model (you can specify other YOLOv8 versions like YOLOv8n, YOLOv8m, etc.)
model = YOLO(r"C:\Users\sahus\VideoAnalytics\yolov8n.pt")  # Start with a pre-trained model, e.g., YOLOv8 Nano

# Global variable to store prediction results
occupied_count = 0
empty_count = 0


@app.route("/", methods=['GET'])
@cross_origin()
def home():

    return render_template('index.html')


@app.route("/train", methods=['POST'])
@cross_origin()
def trainRoute():
    """
    Endpoint to train the YOLOv8 model with the specified data.
    """
    try:
        data_path = request.json.get('data_path', r"C:\Users\sahus\VideoAnalytics\Bench-Detection-1\data.yaml")
        epochs = int(request.json.get('epochs', 1))
        imgsz = int(request.json.get('imgsz', 512))
        batch = int(request.json.get('batch', 4))

        model.train(data=data_path, epochs=epochs, imgsz=imgsz, batch=batch, device="cpu")
        metrics = model.val()

        return jsonify({"message": "Training completed successfully!", "metrics": metrics})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=['POST'])
@cross_origin()

def predictRoute():
    """
    Endpoint to perform predictions on a provided image.
    """
    try:
        image_path = request.json.get('image_path')  # Get the image path from the request

        if not image_path:
            return jsonify({"error": "No image path provided."}), 400

        results = model.predict(source=image_path, save=True)

        # If results are available, process them
        global occupied_count, empty_count
        occupied_count = 0
        empty_count = 0

        if results:
            predictions = results[0].boxes.cls  # Access predictions from the Results object

            for prediction in predictions:
                predicted_class_index = int(prediction.cpu().numpy())  # Convert tensor to int

                # Define a mapping from class index to class name (adjust if necessary)
                class_names = {0: 'Occupied', 3: 'Unoccupied'}

                predicted_class = class_names.get(predicted_class_index, 'Unknown')

                # Increment counts based on predicted classes
                if predicted_class == 'Occupied':
                    occupied_count += 1
                elif predicted_class == 'Unoccupied':
                    empty_count += 1

            return jsonify({
                "message": "Prediction completed successfully!",
                "occupied_count": occupied_count,
                "empty_count": empty_count
            })
        else:
            return jsonify({"error": "No predictions found."}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)  # for AWS
