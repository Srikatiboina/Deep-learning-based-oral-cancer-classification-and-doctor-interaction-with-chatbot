from flask import Flask, request, render_template, jsonify, url_for
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing import image
import io
import random
import nltk
from PIL import Image
nltk.download('punkt')

app = Flask(__name__)

# Load the pre-trained model
model = load_model('mymodel.hdf5')  # Assuming you have saved the model before

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'no file part'})

        file = request.files['file']

        # Load the image
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((224, 224))  # Resize the image to (224, 224)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        print("Array built successfully*********************")
        # Predict
        y_pred = model.predict(img_array)
        print(y_pred)
        y_pred_label = "cancer" if y_pred < 0.05 else "non-cancer"
        print(y_pred_label)

        return render_template('index.html', prediction=y_pred_label)

    return render_template('index.html')


@app.route('/chatbot')
def chatbot():
    bot_response_url = url_for('get_bot_response')  
    return render_template("chatbot.html", bot_response_url=bot_response_url)


@app.route('/get_bot_response', methods=['POST'])
def get_bot_response():
    message = request.form['msg']
    response = ""

    greeting_responses = ["Hello!", "Hi", "Greetings!"]
    goodbye_responses = ["Goodbye!", "See you later!", "Take care!"]

    if message:
        tokens = nltk.word_tokenize(message.lower())

        if any(word in tokens for word in ["hello", "hi"]):
            response = random.choice(greeting_responses)
        elif any(word in tokens for word in ["goodbye", "bye"]):
            response = random.choice(goodbye_responses)
        else:
            oral_cancer = {
                "oralcancer": [
                    "Oral cancer is a type of cancer that affects the mouth, tongue, and lips It is often caused by tobacco and alcohol use, as well as human papillomavirus (HPV) infection Symptoms of oral cancer include mouth sores, difficulty swallowing, and white or red patches on the tongue or mouth."
                ],
                "riskfactors": [
                    "Tobacco and alcohol use are major risk factors for oral cancer.",
                    "HPV infection, particularly HPV-16, is also a significant risk factor.",
                    "Other risk factors include a diet low in fruits and vegetables, and a family history of oral cancer."
                ],
                "symptoms": [
                    "Common symptoms of oral cancer include mouth sores, difficulty swallowing, and white or red patches on the tongue or mouth.",
                    "Other symptoms may include pain or numbness in the mouth, difficulty speaking, and weight loss.",
                    "It is important to see a doctor if you experience any of these symptoms."
                ],
                "diagnosis": [
                    "Oral cancer is typically diagnosed with a biopsy, where a sample of tissue is taken from the affected area.",
                    "Imaging tests such as X-rays, CT scans, and MRI scans may also be used to determine the extent of the cancer.",
                    "A dentist or doctor may also perform a physical examination to look for signs of oral cancer."
                ],
                "treatment": [
                    "Treatment for oral cancer usually involves surgery to remove the tumor, followed by radiation therapy and/or chemotherapy.",
                    "The type of treatment will depend on the stage and location of the cancer, as well as the patient's overall health.",
                    "Early detection and treatment can improve the chances of survival and reduce the risk of recurrence."
                ]
            }

            oral_cancer_list = [w for w in tokens if w in oral_cancer]
            if oral_cancer_list:
                oral_cancer_category = oral_cancer_list[0]
                if oral_cancer_category == "oralcancer":
                    response = random.choice(oral_cancer[oral_cancer_category])
                elif oral_cancer_category == "riskfactors":
                    response = random.choice(oral_cancer[oral_cancer_category])
                elif oral_cancer_category == "symptoms":
                    response = random.choice(oral_cancer[oral_cancer_category])
                elif oral_cancer_category == "diagnosis":
                    response = random.choice(oral_cancer[oral_cancer_category])
                elif oral_cancer_category == "treatment":
                    response = random.choice(oral_cancer[oral_cancer_category])
                else:
                    response = "I'm not sure I understand. Can you please rephrase your question?"
            else:
                response = "Sorry, I don't understand. Please enter a question about oral cancer."

    else:
        response = random.choice(greeting_responses)

    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)