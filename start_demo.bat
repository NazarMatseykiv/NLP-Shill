@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
if not exist data\intents_merged.csv (
  echo Building local dataset...
  python scripts\build_merged_local.py
)
if not exist models\intent_model.joblib (
  echo Training model...
  python train.py
)
echo Starting web demo at http://127.0.0.1:8000
python app.py
