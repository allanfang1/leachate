import numpy as np
import streamlit as st
from google import genai

def llm_explanation(shap_val, prediction_value, target_name):
    client = genai.Client()

    top_features = sorted(
        zip(shap_val.feature_names, shap_val.values, shap_val.data),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:6]    

    features_text = "\n".join([
        f"- {name}: value={val:.2f}, impact={impact:.3f}"
        for name, impact, val in top_features
    ])

    prompt = f"""Predicting leachate given rock properties and event conditions, using SHAP output for explanation:

        Target: {target_name}
        Predicted Value: {prediction_value:.2f}

        SHAP analysis: {features_text}

        Write 3 concise sentences explaining the most influential factors for this prediction for non-technical readers.

        Rules:
            - Base the explanation ONLY on the given feature values and SHAP contributions.
            - Do NOT speculate or claim an effect that contradicts the feature value.
            - Explain the meaning in plain language
    """    
    
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite", contents=prompt
    )
    return response.text


def simple_explanation(shap_val, target_name):
    top_idx = np.argmax(np.abs(shap_val.values))
    feature = shap_val.feature_names[top_idx]
    impact_val = shap_val.values[top_idx]
    feat_val = shap_val.data[top_idx]

    res_dir = "high" if impact_val > 0 else "low"

    if feat_val > 0.1:
        feat_desc = "high"
    elif feat_val < -0.1:
        feat_desc = "low"
    else:
        feat_desc = "average"

    return f"The **{target_name}** is **{res_dir}** mainly due to **{feature}** being **{feat_desc}**."