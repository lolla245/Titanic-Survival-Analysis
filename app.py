from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    pclass = int(request.form["pclass"])
    sex = 0 if request.form["sex"] == "male" else 1
    age = int(request.form["age"])
    fare = int(request.form["fare"])
    family = int(request.form["family"])

    input_data = np.array([[pclass, sex, age, fare, family]])
    result = model.predict(input_data)

    if result[0] == 1:
        msg = "🎉 Survived!"
    else:
        msg = "💀 Did Not Survive"

    return render_template("index.html", prediction=msg)

if __name__ == "__main__":
    app.run(debug=True)