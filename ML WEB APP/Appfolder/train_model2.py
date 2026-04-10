import pandas as pd
# create data set 
data = {
    "study_hours":[7,4,3,2,1,5,6,8,9,10],
    "student_sleep_hours":[5,4,3,9,0,1,2,3,4,5],
    "student_marks_obatained":[40,55,56,60,23,21,20,17,15,10]
}
df = pd.DataFrame(data)
print(df)
from sklearn.linear_model import LinearRegression
import pickle

# Define input (X) and output (y)
X = df[["study_hours", "student_sleep_hours"]]
y = df["student_marks_obatained"]

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Save model
with open("model2.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model 2 trained and saved!")