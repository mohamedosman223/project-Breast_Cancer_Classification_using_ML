from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Load models
models = {
    "linear": joblib.load("linear.joblib"),
    "poly": joblib.load("poly.joblib"),
    "rbf": joblib.load("rbf.joblib"),
    "RF": joblib.load("RF.joblib"),
    "GB": joblib.load("GB.joblib"),
    "KNN": joblib.load("KNN.joblib"),
    "SVC": joblib.load("SVC.joblib")
}

# Initialize imputer
imputer = SimpleImputer(strategy="mean")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Receive data from the client
        data = request.json
        
        # Validate and reshape the features
        features = np.array(data["features"]).reshape(1, -1)
        features_imputed = imputer.fit_transform(features)

        # Get the selected model
        model_name = data.get("model", "linear")
        model = models.get(model_name)

        if not model:
            return jsonify({"error": f"Model '{model_name}' not found!"}), 400

        # Make the prediction
        prediction = model.predict(features_imputed)

        # Log the prediction for debugging
        print(f"Model: {model_name}, Features: {features.tolist()}, Prediction: {prediction[0]}")

        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
