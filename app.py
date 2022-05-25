from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


app = Flask(__name__)

model = load_model("assets/mole_model.h5")

@app.route('/')
def home():
    return render_template('index.html')

        
    
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    img = request.files['mole_img']
    img.save(img.name)
    

    # Load the image
    image_size = (224,224)
    img = image.load_img(img.name, target_size= image_size)
    print('hello')
    # Preprocess the image (as in the training set)
    img_array = image.img_to_array(img)
    img_array = img_array.reshape((1, img_array.shape[0], img_array.shape[1], img_array.shape[2]))
    prepared_img = preprocess_input(img_array)
    # Apply the model to get the prediction
    print('Go for prediction')
    prediction = model.predict(prepared_img)[0]
    print(prediction)
    # Interpret the prediction
    #max = prediction.max()
    idx = int(np.argmax(prediction))
    
    # Display the prediction
    labels = ['nv', 'bkl', 'mel', 'akiec', 'bcc', 'vasc', 'df']
    # You are done ! Congrats :)
    print(idx)
    output = labels[idx]
    

    print(output)

    return render_template('index.html', prediction_text='Mole is type: {}'.format(output))

if __name__ == "__main__":
    app.run(port=5000, debug=True)