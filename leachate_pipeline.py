import pandas as pd
import numpy as np

class LeachatePipeline:
  def __init__(self, xgb_clf, best_models, scaler, categorical_columns, lag_settings, numeric_predictors, feature_columns, leachate_columns, log_targets):
    self.xgb_clf = xgb_clf
    self.best_models = best_models
    self.scaler = scaler
    self.categorical_columns = categorical_columns
    self.lag_settings = lag_settings
    self.numeric_predictors = numeric_predictors
    self.feature_columns = feature_columns
    self.leachate_columns = leachate_columns
    self.log_targets = log_targets
  
  def predict_leachate(self, rock_properties: dict, sequence: list):

    results = []
    processed_rows = [] 

    # Initialize lag features
    lag_values = {f"{col}_lag{lag}": 0 for col in self.leachate_columns + ["Volume_leachate"] for lag in self.lag_settings}

    for i, event in enumerate(sequence):
        # Prepare row
        row = rock_properties.copy()
        row.update(lag_values)

        # Add event features
        for feat in ['Type_event', 'Acid', 'Temp_level']:
            row[feat] = event[feat]

        df_row = pd.DataFrame([row])

        # Scale numeric predictors
        df_row[self.numeric_predictors] = self.scaler.transform(df_row[self.numeric_predictors])

        # One-hot encode event features
        df_row = pd.get_dummies(df_row, columns=['Type_event','Acid','Temp_level'], drop_first=True)

        # Ensure all categorical columns exist
        for col in self.categorical_columns:
            if col not in df_row.columns:
                df_row[col] = 0
        df_row = df_row.reindex(columns=self.feature_columns, fill_value=0)
        processed_rows.append(df_row) 

        # Classifier check
        high_volume_prob = self.xgb_clf.predict_proba(df_row)[:, 1][0]
        high_volume = high_volume_prob >= 0.2  # e.g., 0.5

        # Predict leachate if volume is high
        pred = {}
        pred["Measured"] = 1 if high_volume else 0 
        if high_volume:
            for target in self.leachate_columns:
                model = self.best_models[target]
                pred[target] = model.predict(df_row)[0]
                
                # val = model.predict(df_row)[0]
                # if target in self.log_targets:
                #     val = np.expm1(val)
                # pred[target] = val
        else:
            for target in self.leachate_columns:
                pred[target] = 0  # or np.nan

        # Store prediction
        results.append(pred.copy())

        # Update lag values for next timestep
        for target in self.leachate_columns + ["Volume_leachate"]:
            val = pred[target] if target in pred else 0
            for lag in self.lag_settings:
                lag_values[f"{target}_lag{lag}"] = val if lag == 1 else lag_values[f"{target}_lag{lag-1}"]

    return pd.DataFrame(results), pd.concat(processed_rows, ignore_index=True)