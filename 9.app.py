from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Try common locations automatically
possible_paths = [
    os.path.join(BASE_DIR, "model.pkl"),
    os.path.join(BASE_DIR, "models", "model.pkl"),
    os.path.join(BASE_DIR, "4.models", "model.pkl"),
]

model = None
for path in possible_paths:
    if os.path.exists(path):
        model = pickle.load(open(path, "rb"))
        break

if model is None:
    raise FileNotFoundError(f"model.pkl not found. Checked: {possible_paths}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        pclass = int(request.form["pclass"])
        sex = request.form["sex"]
        age = float(request.form["age"])
        fare = float(request.form["fare"])
        family = int(request.form["family"])

        sex_val = 0 if sex == "male" else 1

        input_data = np.array([[pclass, sex_val, age, fare, family]])

        result = model.predict(input_data)

        prediction = "🎉 Survived!" if result[0] == 1 else "💀 Did Not Survive"

        return render_template("index.html", prediction=prediction)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)