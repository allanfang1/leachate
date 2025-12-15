# [Rock Leachate Predictor](https://roches.streamlit.app/)

A machine learning application that predicts rock leachate characteristics based on rock properties and environmental events.  Built with Streamlit and XGBoost, this tool helps predict both the probability of leaching events and the resulting ion concentrations. 

## Overview

This application uses trained ML models to predict: 
- **Leaching probability**:  Whether a rock sample will produce measurable leachate under specific conditions
- **Ion concentrations**:  Predicted levels of various ions in the leachate when leaching occurs

The predictions are based on rock chemical properties and a sequence of environmental events (rainfall, temperature, acidity).

## Installation

1. Clone the repository:
```bash
git clone https://github.com/allanfang1/leachate.git
cd leachate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure the trained model artifacts are in the `pkl/` directory:
   - `pipeline.pkl` - Contains the trained pipeline with classifier and regression models

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

### Input Parameters

**Rock Properties:**
- EC_rock: Electrical conductivity
- Ph_rock: pH level
- Corg_rock:  Organic carbon content
- SAR_rock: Sodium Adsorption Ratio
- SiO2_rock, Al2O3_rock, Fe2O3_rock, SO3_rock, CaO_rock, MgO_rock: Oxide concentrations

**Environmental Events:**
- Type:  Rain or snow
- Acid: Acid exposure level
- Temp_level: High or low temperature

### How It Works

1. **Input rock properties** and configure environmental events
2. Click **"Predict"** to run the analysis
3. The system first predicts **leaching probability** using a classification model
4. If leaching is predicted, **ion concentrations** are calculated using regression models
5. **SHAP waterfall plots** show feature importance for each prediction
6. Optional **AI explanations** provide natural language insights

## Model Architecture

The `LeachatePipeline` class implements a two-stage prediction system:

1. **Classification Stage**:  XGBoost classifier predicts whether measurable leaching will occur
2. **Regression Stage**: Multiple XGBoost regressors predict individual ion concentrations (only if leaching is predicted)

The pipeline includes:
- Feature scaling for numeric predictors
- One-hot encoding for categorical features
- Lag feature engineering for temporal dependencies
- Log transformation handling for specific target variables
