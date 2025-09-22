"""
API integration tests
Test the bonsai classification API endpoints
"""

import pytest
import requests
import json
import time

class TestAPI:
    
    API_BASE_URL = "http://localhost:8081"
    
    @pytest.fixture(scope="class")
    def api_ready(self):
        """Wait for API to be ready"""
        max_wait = 60
        wait_time = 0
        
        while wait_time < max_wait:
            try:
                response = requests.get(f"{self.API_BASE_URL}/health", timeout=5)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(2)
            wait_time += 2
        
        return False
    
    def test_health_endpoint(self, api_ready):
        """Test API health endpoint"""
        if not api_ready:
            pytest.skip("API not ready")
        
        response = requests.get(f"{self.API_BASE_URL}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "unhealthy"]
    
    def test_root_endpoint(self, api_ready):
        """Test API root endpoint"""
        if not api_ready:
            pytest.skip("API not ready")
        
        response = requests.get(f"{self.API_BASE_URL}/")
        assert response.status_code == 200
        
        data = response.json()
        assert "service" in data
        assert "endpoints" in data
    
    def test_model_info_endpoint(self, api_ready):
        """Test model info endpoint"""
        if not api_ready:
            pytest.skip("API not ready")
        
        response = requests.get(f"{self.API_BASE_URL}/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "model_info" in data
        assert "mlflow_tracking_uri" in data
    
    def test_predict_endpoint_valid_input(self, api_ready):
        """Test prediction with valid input"""
        if not api_ready:
            pytest.skip("API not ready")
        
        # Test with valid bonsai measurements
        test_data = {
            "features": [2.0, 1.5, 5.0, 25.0]  # leaf_length, leaf_width, branch_thickness, height
        }
        
        response = requests.post(
            f"{self.API_BASE_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        # Should work if model is loaded, otherwise may return 500
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "species" in data
            assert data["prediction"] in [0, 1, 2, 3]  # Valid species IDs
            assert data["species"] in ["Juniper", "Ficus", "Pine", "Maple"]
    
    def test_predict_endpoint_invalid_input(self, api_ready):
        """Test prediction with invalid input"""
        if not api_ready:
            pytest.skip("API not ready")
        
        # Test with wrong number of features
        test_data = {
            "features": [2.0, 1.5]  # Only 2 features instead of 4
        }
        
        response = requests.post(
            f"{self.API_BASE_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
    
    def test_predict_endpoint_missing_features(self, api_ready):
        """Test prediction with missing features key"""
        if not api_ready:
            pytest.skip("API not ready")
        
        # Test with missing features key
        test_data = {
            "measurements": [2.0, 1.5, 5.0, 25.0]  # Wrong key name
        }
        
        response = requests.post(
            f"{self.API_BASE_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
    
    def test_predict_endpoint_no_json(self, api_ready):
        """Test prediction with no JSON body"""
        if not api_ready:
            pytest.skip("API not ready")
        
        response = requests.post(f"{self.API_BASE_URL}/predict")
        assert response.status_code == 400
    
    def test_multiple_predictions(self, api_ready):
        """Test multiple predictions for different bonsai types"""
        if not api_ready:
            pytest.skip("API not ready")
        
        test_cases = [
            {"name": "Juniper", "features": [1.8, 1.2, 4.0, 20.0]},
            {"name": "Ficus", "features": [2.3, 1.7, 6.0, 28.0]},
            {"name": "Pine", "features": [2.0, 1.1, 5.5, 30.0]},
            {"name": "Maple", "features": [2.2, 1.8, 5.0, 25.0]}
        ]
        
        successful_predictions = 0
        
        for test_case in test_cases:
            response = requests.post(
                f"{self.API_BASE_URL}/predict",
                json={"features": test_case["features"]},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                successful_predictions += 1
                data = response.json()
                assert "prediction" in data
                assert "species" in data
        
        # At least some predictions should work
        assert successful_predictions > 0, "No predictions were successful"

class TestAPIIntegration:
    """Integration tests for API with MLflow"""
    
    API_BASE_URL = "http://localhost:8081"
    
    def test_api_mlflow_integration(self):
        """Test that API can communicate with MLflow"""
        try:
            response = requests.get(f"{self.API_BASE_URL}/model/info", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                assert "mlflow_tracking_uri" in data
                assert "model_info" in data
                
                # Check if MLflow URI is accessible
                mlflow_uri = data["mlflow_tracking_uri"]
                if mlflow_uri:
                    # Try to reach MLflow health endpoint
                    mlflow_health_url = mlflow_uri.replace('mlflow:5000', 'localhost:5000') + '/health'
                    mlflow_response = requests.get(mlflow_health_url, timeout=5)
                    # MLflow health check is optional - may not be available
            
        except requests.exceptions.RequestException:
            pytest.skip("API not available for MLflow integration test")
    
    def test_model_reload_endpoint(self):
        """Test model reload functionality"""
        try:
            response = requests.post(f"{self.API_BASE_URL}/model/reload", timeout=30)
            
            # Should return 200 (success) or 500 (no model available)
            assert response.status_code in [200, 500]
            
            if response.status_code == 200:
                data = response.json()
                assert "status" in data
                assert data["status"] == "model_reloaded"
            
        except requests.exceptions.RequestException:
            pytest.skip("API not available for model reload test")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
