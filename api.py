import logging
import os
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
import pandas as pd
from pydantic import BaseModel
from function import load_model, train_model, save_model, predict_salary
import uvicorn
from io import StringIO
from dotenv import load_dotenv

load_dotenv()


app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

class InvalidDataError(Exception):
    pass

class ModelNotFoundError(Exception):
    pass

# Custom exception handlers
@app.exception_handler(InvalidDataError)
async def invalid_data_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"message": str(exc)},
    )

@app.exception_handler(ModelNotFoundError)
async def model_not_found_exception_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "Model not found."},
    )

@app.post("/train/")
async def train_model_endpoint(file: UploadFile = File(None), 
                               age: float = Form(None), 
                               gender: str = Form(None),
                               education_level: str = Form(None), 
                               job_title: str = Form(None), 
                               years_of_experience: float = Form(None),
                               filename: str = Form('model.pkl')):
    try:
        if file:
            logger.info("Received file for training.")
            contents = await file.read()
            data = StringIO(contents.decode("utf-8"))
            df = pd.read_csv(data)
        else:
            logger.info("Received form data for training.")
            if None in [age, gender, education_level, job_title, years_of_experience]:
                raise InvalidDataError("All form fields are required.")
            if not isinstance(age, (int, float)) or age < 0:
                raise InvalidDataError("Invalid age value.")
            if not isinstance(years_of_experience, (int, float)) or years_of_experience < 0:
                raise InvalidDataError("Invalid years of experience value.")
            data = {'Age': [age], 'Gender': [gender], 'Education Level': [education_level],
                    'Job Title': [job_title], 'Years of Experience': [years_of_experience]}
            df = pd.DataFrame(data)

        original_row_count = len(df)
        logger.info(f"Original row count: {original_row_count}")

        df_cleaned = df.dropna()
        rows_dropped = original_row_count - len(df_cleaned)
        logger.info(f"Rows dropped due to NaN values: {rows_dropped}")

        logger.info("Training model.")
        model = train_model(df_cleaned)

        logger.info("Saving model.")
        save_model(model)

        return {"message": f"Model trained and saved successfully in the 'model' directory as {filename}. Rows dropped: {rows_dropped}. Rows used for training: {len(df_cleaned)}."}
    except InvalidDataError as e:
        logger.error(f"Invalid data error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

class EmployeeData(BaseModel):
    age: float
    gender: str
    education_level: str
    job_title: str
    years_of_experience: float

    def validate(self):
        if None in [self.age, self.gender, self.education_level, self.job_title, self.years_of_experience]:
            raise InvalidDataError("All fields are required.")
        if not isinstance(self.age, (int, float)) or self.age < 0:
            raise InvalidDataError("Invalid age value.")
        if not isinstance(self.years_of_experience, (int, float)) or self.years_of_experience < 0:
            raise InvalidDataError("Invalid years of experience value.")

JOB_TITLES = [
    "Software Engineer",
    "Web Developer",
    "Data Analyst",
    "UX Designer",
    "Marketing Analyst",
    "Product Manager",
    "Sales Manager",
    "IT Support",
    "Data Scientist",
    "Network Engineer"
]

EDUCATION_LEVELS = ["Bachelor's", "Master's", 'PhD']

@app.post("/predict/")
async def predict_salary_endpoint(employee: EmployeeData):
    try:
        model = load_model()

        employee_dict = {
            "Age": employee.age,
            "Gender": employee.gender,
            "Education Level": employee.education_level,
            "Job Title": employee.job_title,
            "Years of Experience": employee.years_of_experience
        }
        
        df = pd.DataFrame([employee_dict])
        prediction = predict_salary(model, df)
        return {"predicted_salary": prediction[0]}
    except InvalidDataError as e:
        logger.error(f"Invalid data error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=422, detail=str(e))
    except ModelNotFoundError as e:
        logger.error(f"Model not found: {str(e)}", exc_info=True)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/")
async def get_model_response(prompt: str = Query(...)):
    try:
        return call_openai_api(prompt)
    except Exception as e:
        logger.error(f"Error calling external API: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error calling external API")

def call_openai_api(prompt: str):
    headers = {
        'Authorization': f'Bearer {OPENAI_API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data, timeout=60)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.json())
    return response.json()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
