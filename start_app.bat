@echo off
echo Starting Portfolio Construction App...
echo.

echo Upgrading pip...
pip install --upgrade pip

echo.
echo Installing required packages...
pip install -r requirements.txt

echo.
echo Starting Streamlit app...
streamlit run portfolio.py

echo.
echo App closed. Press any key to exit...
pause 