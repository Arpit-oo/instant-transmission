@echo off
cd /d "%~dp0"
if not exist .venv (
  py -3.11 -m venv .venv
  .venv\Scripts\python -m pip install -r requirements.txt
)
.venv\Scripts\python it.py %*
