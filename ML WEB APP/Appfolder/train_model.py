import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Simple dataset
data = {
    "area": [1000, 1500, 2000, 2500, 3000],
    "bedrooms": [2, 3, 3, 4, 4],
    "price": [200000, 300000, 400000, 500000, 600000]
}

df = pd.DataFrame(data)

X = df[["area", "bedrooms"]]
y = df["price"]

model = LinearRegression()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")