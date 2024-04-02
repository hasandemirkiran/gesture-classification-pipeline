from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from typing import List
import logging

import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import json

from typing import Type


from os import listdir
import os
from os.path import isfile, join


def load_data(data_path="data/uWaveGestures.parquet"):
    u_wave_gestures = pd.read_parquet(data_path, engine="pyarrow")

    # drop NAs
    u_wave_gestures = u_wave_gestures.dropna(axis=0, how="any")

    user = u_wave_gestures["user"]

    X = u_wave_gestures.drop(["gesture", "user"], axis=1)
    y = u_wave_gestures["gesture"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return {"user": user, "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


def get_model_version(version):
    if version == "latest":
        models = [f for f in listdir("models") if isfile(join("models", f))]
        latest = ""
        for model in models:
            if latest == "":
                latest = model
            else:
                version = model.split("v")[1].split(".")[0]
                latest_version = latest.split("v")[1].split(".")[0]
                if float(version) > float(latest_version):
                    latest = model
        return join("models", latest)
    else:
        return join("models", f"trained_model_{version}.pkl")


def get_evaluation_version(version):
    if version == "latest":
        reports = [f for f in listdir("reports") if isfile(join("reports", f))]
        latest = ""
        for report in reports:
            if latest == "":
                latest = report
            else:
                version = report.split("v")[1].split(".")[0]
                latest_version = latest.split("v")[1].split(".")[0]
                if float(version) > float(latest_version):
                    latest = report
        return join("reports", latest)
    else:
        return join("reports", f"evaluation_report_{version}.txt")


def get_test_data():
    X_test_1 = pd.read_pickle("test_data/X_test_1.pkl")
    X_test_2 = pd.read_pickle("test_data/X_test_2.pkl")
    X_test_3 = pd.read_pickle("test_data/X_test_3.pkl")
    X_test_4 = pd.read_pickle("test_data/X_test_4.pkl")
    X_test_5 = pd.read_pickle("test_data/X_test_5.pkl")

    return X_test_1, X_test_2, X_test_3, X_test_4, X_test_5


def train_model(user, X_train, X_test, y_train, y_test, version) -> Dict:
    # Subtract 1 from y_train and y_test to ensure zero-based indexing
    y_train_zero_indexed = y_train - 1
    y_test_zero_indexed = y_test - 1

    xgb_clf = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.1)
    xgb_clf.fit(X_train, y_train_zero_indexed)

    xgb_y_predict = xgb_clf.predict(X_test)

    clf_report = classification_report(y_true=y_test_zero_indexed, y_pred=xgb_y_predict, digits=5)
    clf_report_dict = classification_report(
        y_true=y_test_zero_indexed, y_pred=xgb_y_predict, digits=5, output_dict=True
    )

    # save trained model and report locally
    joblib.dump(xgb_clf, f"models/trained_model_{version}.pkl")
    print(clf_report)
    with open(f"reports/evaluation_report_{version}.txt", "w") as report_file:
        report_file.write(clf_report)

    # return evaluation metrics only
    return {"evaluation_metrics": clf_report_dict}


app = FastAPI()


class TrainRequest(BaseModel):
    version: str


class TrainResponse(BaseModel):
    evaluation_metrics: Dict


class PredictionRequest(BaseModel):
    prediction_data: Dict
    version: str


class PredictionResponse(BaseModel):
    predictions: List[int]


class EvaluationRequest(BaseModel):
    version: str


class EvaluationResponse(BaseModel):
    report: str
    model_params: str


@app.get("/train/")
def train_model_api(version: str):
    try:
        # Load data
        train_data = load_data()
        user, X_train, X_test = train_data["user"], train_data["X_train"], train_data["X_test"]
        y_train, y_test = train_data["y_train"], train_data["y_test"]

        # Trigger model training using data from local file
        model_report = train_model(user, X_train, X_test, y_train, y_test, version)

        # Return response with evaluation metrics only
        return TrainResponse(evaluation_metrics=model_report["evaluation_metrics"])

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to train model: {str(e)}")


@app.post("/predict/")
def predict(data: PredictionRequest):
    try:
        model_file_path = get_model_version(data.version)
        model = joblib.load(model_file_path)

        # Perform inference using the trained model
        predictions = model.predict(pd.DataFrame(data.prediction_data))
        return PredictionResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to perform inference: {str(e)}")


@app.get("/get_modelparams_and_report")
def get_modelparams_and_report(version: str):
    try:
        report_file_path = get_evaluation_version(version)
        model_file_path = get_model_version(version)

        f = open(report_file_path, "r")
        report = f.read()
        model = joblib.load(model_file_path)
        return EvaluationResponse(report=report, model_params=str(model.get_params()))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to return model params and evaluation report: {str(e)}")


@app.delete("/delete_model_version")
def delete_model_version(version: str):
    try:
        model_file_path = get_model_version(version)
        if os.path.exists(model_file_path):
            os.remove(model_file_path)

            # Also delete the evaluation report
            report_file_path = get_evaluation_version(version)
            if os.path.exists(report_file_path):
                os.remove(report_file_path)

            return {"message": f"Model version {version} and its evaluation report deleted successfully."}
        else:
            raise HTTPException(status_code=404, detail=f"Model version {version} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model version: {str(e)}")


@app.post("/extend_dataset/")
def extend_dataset(data_path: str):
    try:
        # Load existing data
        existing_data = load_data()
        X_train_existing, X_test_existing = existing_data["X_train"], existing_data["X_test"]

        # Load new data
        new_data = pd.read_pickle(data_path)

        # Drop NAs
        new_data = new_data.dropna(axis=0, how="any")

        # Append new data to existing data
        X_new = new_data.drop(["gesture", "user"], axis=1)
        y_new = new_data["gesture"]
        X_train_extended = pd.concat([X_train_existing, X_new])
        y_train_extended = pd.concat([existing_data["y_train"], y_new])

        # Retrain the model with extended dataset
        model_report = train_model(
            existing_data["user"],
            X_train_extended,
            X_test_existing,
            y_train_extended,
            existing_data["y_test"],
            version="extended",
        )

        # Return response with evaluation metrics only
        return TrainResponse(evaluation_metrics=model_report["evaluation_metrics"])

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extend dataset and retrain model: {str(e)}")
