#!/bin/bash
python /app/src/HRES_ML_Model.py
gunicorn --bind 0.0.0.0:8080 --workers 4 api.recommender_api:app