import pickle
from sklearn.ensemble import RandomForestClassifier
from data_cleaning import clean_data

df = clean_data(r"1.Data/titanic_data.csv")

features = ['Pclass','Sex','Age','Fare','FamilySize']

X = df[features]
y = df['Survived']

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

pickle.dump(model, open(r"4.models/model.pkl", "wb"))

print("Model trained and saved!")