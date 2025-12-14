import numpy as np
import streamlit as st
from google import genai

def llm_explanation(shap_val, prediction_value, target_name):
    client = genai.Client()

    all_features = list(zip(shap_val.feature_names, shap_val.values, shap_val.data))
    
    features_text = "\n".join([
        f"- {name}: value={val:.2f}, impact={impact:.3f}"
        for name, impact, val in all_features
    ])

    prompt = f"""Predicting leachate given rock properties and event conditions, using SHAP values for explanation:

        Target: {target_name}
        Predicted Value: {prediction_value:.2f}

        SHAP analysis: {features_text}

        Write exactly 3 simple sentences explaining why this prediction was made. Focus on the rock properties and conditions. Use plain, accessible language for non computer science experts."""
    
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )
    return response.text


def simple_explanation(shap_val, target_name):
    """
    Naively constructs a sentence based on the top contributing feature.
    Assumes input data is scaled (Z-scores).
    """
    # 1. Find the feature with the biggest impact (positive or negative)
    top_idx = np.argmax(np.abs(shap_val.values))
    feature = shap_val.feature_names[top_idx]
    impact_val = shap_val.values[top_idx]
    feat_val = shap_val.data[top_idx]

    # 2. Determine direction of the result (did this feature push it UP or DOWN?)
    res_dir = "high" if impact_val > 0 else "low"

    # 3. Determine state of the feature (High/Low/Average based on Z-score)
    if feat_val > 0.1:
        feat_desc = "high"
    elif feat_val < -0.1:
        feat_desc = "low"
    else:
        feat_desc = "average"

    return f"The **{target_name}** is **{res_dir}** mainly due to **{feature}** being **{feat_desc}**."