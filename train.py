import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("titanic_data.csv")

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Feature engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Encoding
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Features & target
features = ['Pclass','Sex','Age','Fare','FamilySize']

X = df[features]
y = df['Survived']

# Model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model trained and saved successfully!")