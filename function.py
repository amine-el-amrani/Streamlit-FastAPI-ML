import pickle
import logging
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
import fickling

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(df):
    X = df.drop('Salary', axis=1)
    y = df['Salary']

    # Preprocessing pipeline
    numeric_features = ['Age', 'Years of Experience']
    categorical_features = ['Gender', 'Education Level', 'Job Title']

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='passthrough')

    # Combine preprocessing with the model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor())
    ])

    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [None, 10, 20, 30],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)

    logger.info("Starting model training with hyperparameter tuning.")
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_

    # Cross-validation scores
    scores = cross_val_score(best_model, X, y, cv=5)
    logger.info(f"Cross-validation scores: {scores}")
    logger.info(f"Mean cross-validation score: {scores.mean()}")

    return best_model

def save_model(model):
    model_directory = 'model'
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    
    # Generate a unique filename with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_{timestamp}.pkl"
    model_path = os.path.join(model_directory, filename)

    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

    return filename

def load_model():
    model_directory = 'model'

    model_files = [f for f in os.listdir(model_directory)]
    if not model_files:
        raise FileNotFoundError("No models found in the 'model' directory.")
    
    latest_model_file = max(model_files, key=lambda x: os.path.getctime(os.path.join(model_directory, x)))
    model_path = os.path.join(model_directory, latest_model_file)

    with open(model_path, 'rb') as file:
        return fickling.load(file)

def predict_salary(model, X):
    return model.predict(X)
