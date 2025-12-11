# critical-heat-flux-ml
Machine-learning framework for predicting Critical Heat Flux (CHF) using XGBoost and LightGBM, based on 184 data points from 21 pool-boiling studies.

# ============================================
# Virtual Environment Setup (Python 3.10+)
# ============================================

# 1) Create a virtual environment called "venv"
python3 -m venv venv

# 2) Activate the environment
# ---- Windows (PowerShell) ----
# .\venv\Scripts\Activate.ps1
#
# ---- Windows (cmd.exe) ----
# venv\Scripts\activate.bat
#
# ---- macOS/Linux ----
# source venv/bin/activate

# 3) Upgrade pip
pip install --upgrade pip

# 4) Install required packages
pip install -r requirements.txt

# 5) (Optional) Confirm installation
python -c "import xgboost, lightgbm, sklearn, shap; print('Environment OK')"
