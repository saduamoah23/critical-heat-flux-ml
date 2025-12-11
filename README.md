# critical-heat-flux-ml

Machine-learning framework for predicting Critical Heat Flux (CHF) using XGBoost and LightGBM, based on 184 data points from 21 pool-boiling studies.

## Virtual Environment Setup (Python 3.10+)

### 1) Create a virtual environment called "venv"

```bash
python3 -m venv venv
```

### 2) Activate the environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (cmd.exe):**
```cmd
venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 3) Upgrade pip

```bash
pip install --upgrade pip
```

### 4) Install required packages

```bash
pip install -r requirements.txt
```

### 5) (Optional) Confirm installation

```bash
python -c "import xgboost, lightgbm, sklearn, shap; print('Environment OK')"
```
