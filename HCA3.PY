import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Sample dataset of symptoms and conditions
data = {
    "Headache": [1, 1, 0, 0, 1, 0],
    "Fever": [1, 1, 0, 0, 1, 0],
    "Fatigue": [1, 1, 1, 0, 1, 0],
    "Cough": [1, 1, 1, 0, 1, 0],
    "Nausea": [1, 0, 0, 1, 1, 0],
    "Shortness_of_Breath": [0, 1, 0, 0, 0, 0],
}

# Conditions corresponding to the symptoms
conditions = ['Cold', 'Flu', 'Diabetes', 'Cold', 'Flu', 'Cold']

# Convert data into DataFrame
df = pd.DataFrame(data)
df['Condition'] = conditions

# Label encode the conditions into numeric labels
label_encoder = LabelEncoder()
df['Condition'] = label_encoder.fit_transform(df['Condition'])

# Split the data into features (X) and target (y)
X = df.drop('Condition', axis=1)
y = df['Condition']

# Train a Decision Tree Classifier model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Function to predict health condition based on symptoms
def predict_condition(symptoms):
    symptoms_input = np.array([symptoms])
    prediction = model.predict(symptoms_input)
    predicted_condition = label_encoder.inverse_transform(prediction)
    return predicted_condition[0]

# Conversational AI Assistant with symptoms-based questions
def conversational_assistant():
    print("Hello! I'm your Healthcare AI Assistant. Let's talk about your symptoms.")
    
    # Symptom questions (yes = 1, no = 0)
    symptoms = {
        "Headache": "Do you have a headache? (yes/no): ",
        "Fever": "Do you have a fever? (yes/no): ",
        "Fatigue": "Do you feel fatigued? (yes/no): ",
        "Cough": "Do you have a cough? (yes/no): ",
        "Nausea": "Do you feel nauseous? (yes/no): ",
        "Shortness_of_Breath": "Are you experiencing shortness of breath? (yes/no): "
    }

    # Ask the user about their symptoms
    user_symptoms = []
    for symptom, question in symptoms.items():
        while True:
            answer = input(question).strip().lower()
            if answer == 'yes':
                user_symptoms.append(1)
                break
            elif answer == 'no':
                user_symptoms.append(0)
                break
            else:
                print("Please answer with 'yes' or 'no'.")
    
    # Get the prediction
    result = predict_condition(user_symptoms)
    print(f"\nBased on your symptoms, it seems you may have: {result}.")

    # Ask if the user wants to check another person
    again = input("\nWould you like to check another person? (yes/no): ").strip().lower()
    if again == 'yes':
        conversational_assistant()
    else:
        print("\nThank you for using the Healthcare Assistant! Stay healthy!")

if __name__ == "__main__":
    conversational_assistant()
