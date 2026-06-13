import pickle
import pandas as pd

model = pickle.load(open(r"4.models/model.pkl", "rb"))

sample = pd.DataFrame([{
    "Pclass": 3,
    "Sex": 0,
    "Age": 22,
    "Fare": 7.25,
    "FamilySize": 1
}])

prediction = model.predict(sample)

print("Survived" if prediction[0] == 1 else "Not Survived")