# Salary Prediction System

This project is a salary prediction system built with FastAPI and Streamlit. It allows users to train a machine learning model on salary data, make predictions, and interact with a language model provided by OpenAI.

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### 1. Clone the Repository

```bash
git clone https://github.com/amine-el-amrani/Streamlit-FastAPI-ML.git
cd Streamlit-FastAPI-ML
```

### 2. Create and activate the Virtual Environment
Create a virtual environment to manage dependencies

```bash
python3 -m venv venv
```

Activate the virtual environment:
```bash
python3 -m venv venv
```
### 3. Install Requirements
Install the necessary dependencies using pip.

```bash
pip install -r requirements.txt
```

### 4.Set Up Environment Variables
Create a .env file in the project root directory and add your OpenAI API key:

```bash
OPENAI_API_KEY=your_openai_api_key
```
Replace your_openai_api_key with your actual OpenAI API key.

### 5.Run the FastAPI Server
Start the FastAPI server:

```bash
uvicorn api:app --reload
```
The server will be running at http://127.0.0.1:8000.

### 6.Run the Streamlit Application
In a new terminal (with the virtual environment activated), run the Streamlit app:

```bash
streamlit run app.py
```
The Streamlit interface will open in your default web browser.

## 7.Project Structure
 
```bash
│
├── .env                 # Environment variables file
├── .gitignore           # Git ignore file
├── README.md            # Project documentation
├── api.py               # FastAPI application
├── app.py               # Streamlit application
├── function.py          # Model training and prediction functions
├── requirements.txt     # Project dependencies
├── model/               # Directory to save trained models
```

## API Endpoints

### 1.Train Model
- Endpoint: /train/
- Method: POST
- Description: Trains a model on the provided data.
- Request Parameters (Form or File Upload)

### 2.Predict Salary
- Endpoint: /predict/
- Method: POST
- Description: Predicts the salary based on the provided employee data.
- Request Body (JSON)
{
  "age": 30,
  "gender": "Male",
  "education_level": "Master's",
  "job_title": "Software Engineer",
  "years_of_experience": 5
}

### 3.Interact with Language Model
- Endpoint: /model/
- Method: GET
- Description: Interacts with the OpenAI language model using the provided prompt.
- Query Parameters: prompt: The prompt to send to the language model.