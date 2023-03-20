import streamlit as st
from PIL import Image
import numpy as np
from tensorflow import keras

options = st.sidebar.radio('',options=['Home','Predict'])
model = keras.models.load_model('ResNetModel')
class_names = ['Healthy', 'Tumor']

if options == 'Home':
    healthy = Image.open('data/Healthy/Not Cancer  (1).jpeg').resize((180, 180))
    tumor = Image.open('data/BrainTumor/Cancer (1).png').resize((180, 180))

    st.markdown('Brain Tumor Prediction')
    st.markdown("Brain tumors are a devastating disease that can have a significant impact on patients' physical, emotional, and financial well-being." +
    "With brain tumors being the leading cause of cancer-related deaths in children under 14, and approximately 700,000 people living with a brain tumor" +
    "in the United States alone, the need for effective diagnosis and treatment is critical. Early detection is key, as the overall 5-year survival rate" + 
    "for all primary malignant brain tumors is only around 36\%. In this paper, we investigate the use of machine learning models, including logistic regression," +
    "multi-layer perceptron, convolutional neural networks, and even a pre-trained RestNet network model to predict brain tumors from magnetic resonance imaging (MRI) data.")
    st.markdown('Data Information')
    st.write('  -     X Healthy Images')
    st.image(healthy)
    st.write('  -     X Tumor Images')
    st.image(tumor)

if options == 'Predict':
    uploaded_file = st.file_uploader('Upload an MRI scan of a brain', type=['png', 'jpg', 'jpeg', 'tif'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image)
        image = image.convert('RGB').resize((180, 180))
        image = np.expand_dims(image, axis=0)
        pred = model.predict(image)
        output_class = class_names[np.argmax(pred)]
        st.markdown(f'Prediction: {output_class}')
