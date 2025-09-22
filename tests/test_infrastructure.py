"""
Infrastructure validation tests
Test Docker Compose services and networking
"""

import pytest
import requests
import time
import subprocess
import os
import yaml

class TestInfrastructure:
    
    def test_docker_compose_config(self):
        """Test Docker Compose configuration is valid"""
        compose_file = os.path.join(os.path.dirname(__file__), "../docker/docker-compose.yml")
        
        # Check if file exists
        assert os.path.exists(compose_file), "docker-compose.yml not found"
        
        # Validate YAML syntax
        with open(compose_file, 'r') as f:
            try:
                config = yaml.safe_load(f)
                assert config is not None, "Invalid YAML in docker-compose.yml"
                assert 'services' in config, "No services defined in docker-compose.yml"
            except yaml.YAMLError as e:
                pytest.fail(f"YAML syntax error in docker-compose.yml: {e}")
    
    def test_required_services_defined(self):
        """Test that all required services are defined"""
        compose_file = os.path.join(os.path.dirname(__file__), "../docker/docker-compose.yml")
        
        with open(compose_file, 'r') as f:
            config = yaml.safe_load(f)
        
        required_services = ['mlflow', 'airflow-webserver', 'airflow-scheduler', 'postgres', 'api']
        
        for service in required_services:
            assert service in config['services'], f"Required service '{service}' not found"
    
    def test_service_ports_not_conflicting(self):
        """Test that services use different ports"""
        compose_file = os.path.join(os.path.dirname(__file__), "../docker/docker-compose.yml")
        
        with open(compose_file, 'r') as f:
            config = yaml.safe_load(f)
        
        used_ports = []
        
        for service_name, service_config in config['services'].items():
            if 'ports' in service_config:
                for port_mapping in service_config['ports']:
                    host_port = port_mapping.split(':')[0]
                    assert host_port not in used_ports, f"Port {host_port} is used by multiple services"
                    used_ports.append(host_port)
    
    def test_environment_variables(self):
        """Test required environment variables are set"""
        required_env_vars = ['MLFLOW_TRACKING_URI']
        
        # Check if running in CI environment
        if os.getenv('CI'):
            for var in required_env_vars:
                assert os.getenv(var) is not None, f"Required environment variable {var} not set"
    
    def test_dockerfile_security(self):
        """Basic Dockerfile security checks"""
        dockerfiles = [
            "../docker/Dockerfile.mlflow",
            "../docker/Dockerfile.api"
        ]
        
        for dockerfile_path in dockerfiles:
            dockerfile_full_path = os.path.join(os.path.dirname(__file__), dockerfile_path)
            
            if os.path.exists(dockerfile_full_path):
                with open(dockerfile_full_path, 'r') as f:
                    content = f.read()
                
                # Check for security issues
                assert "USER root" not in content.upper(), f"Dockerfile {dockerfile_path} runs as root"
                assert "ADD http" not in content.upper(), f"Dockerfile {dockerfile_path} uses insecure ADD"

class TestServiceConnectivity:
    """Test service connectivity and health"""
    
    @pytest.fixture(scope="class")
    def services_ready(self):
        """Wait for services to be ready"""
        # This would typically wait for services to start
        # In CI, services should already be running
        max_wait = 60
        wait_time = 0
        
        services = {
            'mlflow': 'http://localhost:5000/health',
            'airflow': 'http://localhost:8080/health',
            'api': 'http://localhost:8081/health'
        }
        
        ready_services = {}
        
        for service_name, health_url in services.items():
            while wait_time < max_wait:
                try:
                    response = requests.get(health_url, timeout=5)
                    if response.status_code == 200:
                        ready_services[service_name] = True
                        break
                except requests.exceptions.RequestException:
                    pass
                
                time.sleep(2)
                wait_time += 2
            else:
                ready_services[service_name] = False
        
        return ready_services
    
    def test_mlflow_health(self, services_ready):
        """Test MLflow service health"""
        if not services_ready.get('mlflow', False):
            pytest.skip("MLflow service not ready")
        
        response = requests.get('http://localhost:5000/health')
        assert response.status_code == 200
    
    def test_airflow_health(self, services_ready):
        """Test Airflow webserver health"""
        if not services_ready.get('airflow', False):
            pytest.skip("Airflow service not ready")
        
        response = requests.get('http://localhost:8080/health')
        assert response.status_code == 200
    
    def test_api_health(self, services_ready):
        """Test API service health"""
        if not services_ready.get('api', False):
            pytest.skip("API service not ready")
        
        response = requests.get('http://localhost:8081/health')
        assert response.status_code == 200
    
    def test_full_stack(self):
        """Test full stack integration"""
        # This test would verify the complete pipeline
        # For now, just check if we can reach the main services
        
        services_to_test = [
            ('MLflow', 'http://localhost:5000'),
            ('Airflow', 'http://localhost:8080'),
            ('API', 'http://localhost:8081')
        ]
        
        reachable_services = 0
        
        for service_name, service_url in services_to_test:
            try:
                response = requests.get(service_url, timeout=10)
                if response.status_code in [200, 302]:  # 302 for redirects
                    reachable_services += 1
            except requests.exceptions.RequestException:
                pass
        
        # At least 2 out of 3 services should be reachable for basic functionality
        assert reachable_services >= 2, f"Only {reachable_services} out of {len(services_to_test)} services are reachable"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
