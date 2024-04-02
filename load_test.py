from locust import HttpUser, TaskSet, task, between
import pandas as pd


class MyTaskSet(TaskSet):
    @task
    def predict(self):
        test_data = pd.read_pickle("test_data/X_test_1.pkl")
        data = {
            "prediction_data": test_data.to_dict(),
            "version": "latest",
        }
        self.client.post("/predict/", json=data)


class WebsiteUser(HttpUser):
    tasks = [MyTaskSet]
    wait_time = between(1, 5)  # Adjust wait time as needed
