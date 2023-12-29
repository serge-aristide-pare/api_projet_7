import os
import unittest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from main import app
from joblib import load
import lightgbm as lgbm
from lightgbm import LGBMClassifier
from lightgbm import plot_importance

client = TestClient(app)

class TestCreditScoringAPI(unittest.TestCase):
    
    def test_predict_endpoint(self):
        response = client.post('/predict', json={"id_client": 263589})
        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        self.assertIn("client_id", json_response)
        self.assertIn("prediction", json_response)
        self.assertIn("probabilite", json_response)

    def test_root_endpoint(self):
        response = client.get('/Loan application scoring dashboard')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'message': 'Bienvenu dans notre tableau de bord du credit scoring !'})

    def test_shap_values_local_endpoint(self):
        response = client.get('/shaplocal/263589')
        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        self.assertIn("shap_values", json_response)
        self.assertIn("base_value", json_response)
        self.assertIn("data", json_response)
        self.assertIn("feature_names", json_response)

    def test_shap_values_endpoint(self):
        response = client.get('/shap')
        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        self.assertIn("shap_values_0", json_response)
        self.assertIn("shap_values_1", json_response)
        self.assertIn("feature_names", json_response)

if __name__ == '__main__':
    unittest.main()
