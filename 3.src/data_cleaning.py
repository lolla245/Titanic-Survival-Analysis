import pandas as pd

def clean_data(path):
    df = pd.read_csv(path)

    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    df.drop(columns=['Cabin'], inplace=True)

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    return df