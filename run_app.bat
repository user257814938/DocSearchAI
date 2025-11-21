@echo off
set PYTHONPATH=%PYTHONPATH%;C:\tmp\libs
python -m streamlit run app.py
pause
