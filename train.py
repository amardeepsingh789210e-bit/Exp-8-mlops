import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df = pd.read_csv('Iris.csv')
df = df.drop("Id",axis=1)
x = df.drop("Species",axis=1)
y = df["Species"]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")
joblib.dump(model, 'model.pkl')
print("Model saved as model.pkl")

