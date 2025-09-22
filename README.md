
# HRES ESG Recommender System - MLOps PhD Prototype

This repository contains the complete MLOps stack for a PhD research prototype of a Hybrid Renewable Energy System (HRES) recommender. It is designed for automation, reproducibility, and governance, integrating Environmental, Social, and Governance (ESG) criteria into its core.

---

## 🚀 Getting Started: 5-Minute Quick Start

Follow these steps to get the entire system up and running on your local machine.

### Step 1: Prerequisites

-   **Docker & Docker Compose**: Ensure you have the latest versions installed.
-   **Git**: For cloning the repository.
-   **Azure OpenAI Credentials (Optional)**: Required for the AI Advisor chat feature.

### Step 2: Clone the Repository

Clone this project to your local machine:
```bash
git clone <your-repository-url>
cd HRES_ESG_Project
```

### Step 3: Configure Your Environment

All configuration is handled in a single `.env` file.

1.  Navigate to the `docker/` directory.
2.  Copy the example environment file:
    ```bash
    cd docker
    cp .env.example .env
    ```
3.  **Edit the `docker/.env` file:**
    -   **`AZURE_OPENAI_API_KEY`**: **(IMPORTANT)** Replace the placeholder `"YOUR_AZURE_OPENAI_API_KEY_HERE"` with your actual key. If you don't have one, the chat feature will be disabled, but the rest of the application will work.
    -   All other variables (like database passwords) can be left as their default values for local development.

### Step 4: Build and Run the System

From the `docker/` directory, run the following command to build all the container images and start the services:

```bash
docker compose up --build -d
```
-   `--build`: Rebuilds the images to include any code changes.
-   `-d`: Runs the containers in detached (background) mode.

The initial build may take several minutes as it downloads base images and installs dependencies.

### Step 5: Access the Services

Once the containers are running (check status with `docker compose ps`), you can access the services in your browser:

-   🌐 **User Interface (Streamlit)**: [http://localhost:8501](http://localhost:8501)
-   🖥️ **Airflow Webserver**: [http://localhost:8080](http://localhost:8080) (Login: `airflow` / `airflow`)
-   📊 **MLflow Tracking Server**: [http://localhost:5000](http://localhost:5000)
-   🧪 **JupyterLab Notebooks**: [http://localhost:8888](http://localhost:8888)

---

## ⚙️ Core Workflow: From Zero to Recommendation

After starting the system, the database and models are empty. You must perform this crucial first step.

### 1. Initial Data & Model Generation (Crucial First Step)

You need to run the automation pipeline once to generate the dataset and train the initial models.

1.  **Go to the Airflow UI**: [http://localhost:8080](http://localhost:8080)
2.  **Un-pause the DAG**: Find `HRES_PhD_Automation_Pipeline` and click the toggle button to enable it.
3.  **Trigger the DAG**: Click the "Play" button (▶️) on the right to start a new DAG run.
4.  **Monitor the Run**: Click on the DAG name and watch the "Graph" or "Grid" view as the tasks execute. This run will:
    -   Generate `HRES_Dataset.csv`.
    -   Train the ML models and register them in MLflow.
    -   Run validation tests.

### 2. Use the Application

Once the Airflow DAG run is successful, the main application is ready to use.
-   **Go to the UI**: [http://localhost:8501](http://localhost:8501)
-   The "ESG Recommender" and "ML Fast Predictor" tabs will now be fully functional.

### 3. Experiment and Explore

-   **Go to JupyterLab**: [http://localhost:8888](http://localhost:8888)
-   Navigate to the `notebooks/` directory and open `1_HRES_Experimentation.ipynb`.
-   Here you can perform ad-hoc analysis, test new scenarios, and manually log experiments to MLflow.

---

## 🧪 Automated Testing

To verify the integrity of the system's components, you can run the automated tests. Ensure the stack is running first (`docker compose up -d`).

From the `docker/` directory, run:
```bash
docker compose run --rm hres_api pytest tests/
```
This command starts a temporary container with all dependencies and executes the test suite.

---

## 🏗️ System Architecture

-   **`hres_api`**: A Flask & Gunicorn backend serving the core logic.
-   **`hres_ui`**: A Streamlit web application providing the user interface.
-   **`hres_airflow`**: Apache Airflow for orchestrating automation pipelines.
-   **`hres_mlflow`**: An MLflow Tracking Server for experiment tracking and model registry.
-   **`hres_jupyter`**: A JupyterLab environment for interactive experimentation.
-   **`hres_postgres`**: A PostgreSQL database for Airflow and MLflow metadata.

---

## ⏹️ Managing the System

All commands should be run from the `docker/` directory.

-   **Check container status**:
    ```bash
    docker compose ps
    ```
-   **View logs for a specific service**:
    ```bash
    docker compose logs -f hres_api
    ```
-   **Stop all services**:
    ```bash
    docker compose down
    ```
-   **Stop services and remove all data volumes** (database, MLflow artifacts):
    ```bash
    docker compose down -v
    ```
```