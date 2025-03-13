@echo off
REM --- 가상환경 활성화 ---
call venv\Scripts\activate

REM --- 스크립트 실행 ---
python my_script.py

REM --- 종료 전에 결과 확인 ---
pause
