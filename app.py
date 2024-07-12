import os
import streamlit as st
import requests
from api import JOB_TITLES, EDUCATION_LEVELS
from security import safe_requests

st.title('Salary Prediction System')

st.header("Model Training")
train_method = st.radio("Select training input method:", ("Upload CSV File", "Manual Input"))
if train_method == "Upload CSV File":
    train_file = st.file_uploader("Upload CSV file to train the model", type="csv")
    if st.button("Train with CSV"):
        train_response = requests.post("http://127.0.0.1:8000/train/", files={"file": train_file.getvalue()})
        if train_response.ok:
            st.success("Model trained successfully!")
        else:
            st.error("Failed to train model.")
elif train_method == "Manual Input":
    with st.form(key='training_form'):
        age = st.number_input('Age', min_value=18, max_value=65)
        gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
        education_level = st.selectbox('Education Level', EDUCATION_LEVELS)
        job_title = st.selectbox('Job Title', JOB_TITLES)
        years_of_experience = st.number_input('Years of Experience', min_value=0, max_value=50)
        submit_train = st.form_submit_button("Train Model")
        if submit_train:
            response = requests.post("http://127.0.0.1:8000/train/", data={
                'age': age, 'gender': gender, 'education_level': education_level,
                'job_title': job_title, 'years_of_experience': years_of_experience
            })
            if response.ok:
                st.success("Model trained successfully!")
            else:
                st.error("Failed to train model.")

st.header("Salary Prediction")
with st.form(key='predict_form'):
    age = st.number_input('Age', min_value=18, max_value=65)
    gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
    education_level = st.selectbox('Education Level', EDUCATION_LEVELS)
    job_title = st.selectbox('Job Title', JOB_TITLES)
    years_of_experience = st.number_input('Years of Experience', min_value=0, max_value=50)
    submit_predict = st.form_submit_button("Predict Salary")

    if submit_predict:
        data = {
            'age': age,
            'gender': gender,
            'education_level': education_level,
            'job_title': job_title,
            'years_of_experience': years_of_experience
        }
        response = requests.post("http://127.0.0.1:8000/predict/", json=data)
        if response.status_code == 200:
            predicted_salary = response.json()['predicted_salary']
            st.success(f"The predicted salary is: ${predicted_salary:.2f}")
        else:
            st.error(f"Failed to get prediction: {response.text}")


def list_model_files(directory='model'):
    """ Lists all files in the specified directory. """
    files = [f for f in os.listdir(directory)]
    return files

def get_file_path(filename):
    """ Returns the file path of the model in the 'model/' directory. """
    return os.path.join('model', filename)

st.title('Download Models')

model_files = list_model_files()
model_file = st.selectbox("Select a model to download:", model_files)

file_path = get_file_path(model_file)
with open(file_path, "rb") as fp:
    st.download_button(
        label="Download Model",
        data=fp,
        file_name=model_file,
        mime='application/octet-stream'
    )


st.header("Interact with Language Model")

def get_model_response(prompt):
    url = f"http://127.0.0.1:8000/model/"
    params = {"prompt": prompt}
    response = safe_requests.get(url, params=params)
    return response

with st.form(key='model_form'):
    prompt = st.text_area('Enter your prompt')
    submit_model = st.form_submit_button("Get Model Response")

    if submit_model:
        response = get_model_response(prompt)
        if response.status_code == 200:
            model_output = response.json()
            st.success("Model response received successfully!")
            st.write(model_output)
        else:
            st.error(f"Failed to get model response: {response.text}")
