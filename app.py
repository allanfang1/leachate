import streamlit as st
import joblib
from leachate_pipeline import LeachatePipeline
from explainer import simple_explanation, llm_explanation
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# @st.cache_resource 
def load_artifacts():
    pipeline = joblib.load("./pkl/pipeline.pkl")

    return pipeline

pipeline = load_artifacts()
leachate_columns = list(pipeline.best_models.keys())

st.title("Rock Leachate Predictor")

if "predict_clicked" not in st.session_state:
    st.session_state["predict_clicked"] = False

if st.button("Predict", key="predict_top", on_click=lambda: st.session_state.update({"predict_clicked": True})):
    pass

with st.expander("Input Rock Properties", expanded=True):
    col_rock, col_events = st.columns(2)
    with col_rock:
        rock_input = {
            "EC_rock": st.number_input("EC Rock", 2.3),
            "Ph_rock": st.number_input("Ph Rock", 6.5),
            "Corg_rock": st.number_input("Corg Rock", 1.2),
            "SAR_rock": st.number_input("SAR Rock", 0.8),
            "SiO2_rock": st.number_input("SiO2 Rock", 45.0),
            "Al2O3_rock": st.number_input("Al2O3 Rock", 15.0),
            "Fe2O3_rock": st.number_input("Fe2O3 Rock", 5.0),
            "SO3_rock": st.number_input("SO3 Rock", 0.2),
            "CaO_rock": st.number_input("CaO Rock", 8.0),
            "MgO_rock": st.number_input("MgO Rock", 3.0),
        }

    st.divider()
    with col_events:
        num_events = st.slider("How many events?", 1, 10, 1)
        events_list = []

        for i in range(num_events):
            st.subheader(f"Event {i+1}")
            col1, col2, col3 = st.columns(3)
            with col1:
                e_type = st.selectbox(f"Type {i+1}", ["rain", "snow"])
            with col2:
                acid = st.number_input(f"Acid {i+1}", value=0)
            with col3:
                temp = st.selectbox(f"Temp_level {i+1}", ["high", "low"])
            
            events_list.append({
                "Type_event": e_type,
                "Acid": acid,
                "Temp_level": temp,
            })

if st.session_state.get("predict_clicked", False):
    print("Button clicked")

    # rock_input = {
    # 'EC_rock': 1351, 'Ph_rock': 8.11, 'Corg_rock': 0.05, 'SAR_rock': 0.04,
    # 'SiO2_rock': 21.78, 'Al2O3_rock': 7.38, 'Fe2O3_rock': 3.16, 'SO3_rock': 24.65,
    # 'CaO_rock': 20.55, 'MgO_rock': 2.88
    # }

    # events_list = [
    # {'Type_event':'rain', 'Acid':0, 'Temp_level':'low'},
    # {'Type_event':'rain', 'Acid':1, 'Temp_level':'low'},
    # {'Type_event':'rain', 'Acid':0, 'Temp_level':'high'},
    # ]

    print(pipeline.log_targets)
    raw_predictions, df_proc = pipeline.predict_leachate(rock_input, events_list)
    # print(rock_input)
    # print(events_list)
    print(raw_predictions)

    with st.expander("Results", expanded=True):
        st.write("Predictions")
        st.dataframe(raw_predictions)
        st.write("The below \"Waterfall\" plots display explanations for individual predictions. The bottom of a waterfall plot starts as the expected value of the model output, and then each row shows how the positive (red) or negative (blue) contribution of each feature moves the value from the expected model output to the model output for this prediction.")

        st.divider()

        X_shap = df_proc[pipeline.feature_columns].fillna(-999)

        explainer_clf = shap.TreeExplainer(pipeline.xgb_clf, model_output='raw')
        shap_values_clf = explainer_clf(X_shap)

        for i in range(len(X_shap)):
            st.subheader(f"Event {i+1}")
            st.markdown("**Leaching Probability**")
            prob = pipeline.xgb_clf.predict_proba(X_shap.iloc[[i]])[0, 1]
            st.metric("Probability", f"{prob:.2%}")
            # st.info(simple_explanation(shap_values_clf[i], "Chance of leaching"))
            # st.info(llm_explanation(shap_values_clf[i], prob, "Chance of leaching"))  
            if st.button("Get explanation", key=f"explain_prob_{i}"):
                with st.spinner("Generating..."):
                    st.info(llm_explanation(shap_values_clf[i], prob, "Chance of leaching"))

            fig, ax = plt.subplots(figsize=(6, 4))
            shap.plots.waterfall(shap_values_clf[i], show=False, max_display=7)
            st.pyplot(fig)
            plt.close(fig)

            if raw_predictions.iloc[i]["Measured"] == 1:
                st.markdown("#### Ion Concentrations")
                # Use tabs to organize the many ions
                ion_tabs = st.tabs(leachate_columns)
                
                for idx, ion in enumerate(leachate_columns):
                    with ion_tabs[idx]:
                        # Calculate SHAP for this specific event and ion
                        reg_explainer = shap.TreeExplainer(pipeline.best_models[ion], model_output='raw')
                        shap_values_reg = reg_explainer(X_shap.iloc[[i]])
                        
                        fig_r, ax_r = plt.subplots(figsize=(6, 4))
                        shap.plots.waterfall(shap_values_reg[0], show=False, max_display=7)
                        st.pyplot(fig_r)
                        plt.close(fig_r)

                        val = raw_predictions.iloc[i][ion]
                        st.metric(f"{ion} Value", f"{val:.2f}")

                        # st.info(simple_explanation(shap_values_reg[0], f"{ion} level"))
                        # st.info(llm_explanation(shap_values_reg[0], val, f"{ion} level"))
                        if st.button("Get explanation", key=f"explain_{i}_{ion}"):
                            with st.spinner("Generating..."):
                                st.info(llm_explanation(shap_values_reg[0], val, f"{ion} level"))
            else:
                st.caption("Predicted little to no leaching")

# import streamlit as st
# import joblib
# from explainer import simple_explanation
# import pandas as pd
# import numpy as np
# import shap
# import matplotlib.pyplot as plt

# @st.cache_resource 
# def load_artifacts():
#     clf = joblib.load("./pkl/xgb_classifier.pkl")
#     regs = joblib.load("./pkl/xgb_regressors.pkl")
#     scaler = joblib.load("./pkl/scaler.pkl")

#     feature_columns = joblib.load("./pkl/feature_columns.pkl")
#     categorical_columns = joblib.load("./pkl/categorical_columns.pkl")
#     lag_settings = joblib.load("./pkl/lag_settings.pkl")
#     log_targets = [
#         'EC_leachate', 'Chloride_leachate', 'Carbonate_leachate',
#         'Sulfate_leachate', 'Nitrate_leachate', 'Phosphate_leachate',
#         'Ca_leachate', 'Fe_leachate', 'K_leachate',
#         'Mg_leachate', 'Mn_leachate', 'Na_leachate'
#     ]

#     return clf, regs, scaler, feature_columns, lag_settings, log_targets, categorical_columns

# clf, regs, scaler, feature_columns, lag_settings, log_targets, categorical_columns = load_artifacts()
# leachate_columns = list(regs.keys())

# def predict_leachate(rock_properties: dict, sequence: list) -> pd.DataFrame:

#     numeric_predictors = ['EC_rock', 'Ph_rock', 'Corg_rock', 'SAR_rock',
#                       'SiO2_rock', 'Al2O3_rock', 'Fe2O3_rock', 'SO3_rock',
#                       'CaO_rock', 'MgO_rock']

#     results = []
#     processed_rows = []

#     # Initialize lag features
#     lag_values = {f"{col}_lag{lag}": 0 for col in leachate_columns + ["Volume_leachate"] for lag in lag_settings}

#     for i, event in enumerate(sequence):
#         # Prepare row
#         row = rock_properties.copy()
#         row.update(lag_values)

#         # Add event features
#         for feat in ['Type_event', 'Acid', 'Temp_level']:
#             row[feat] = event[feat]

#         df_row = pd.DataFrame([row])

#         # Scale numeric predictors
#         df_row[numeric_predictors] = scaler.transform(df_row[numeric_predictors])

#         # One-hot encode event features
#         df_row = pd.get_dummies(df_row, columns=['Type_event','Acid','Temp_level'], drop_first=True)

#         # Ensure all categorical columns exist
#         for col in categorical_columns:
#             if col not in df_row.columns:
#                 df_row[col] = 0
#         df_row = df_row.reindex(columns=feature_columns, fill_value=0)

#         print(df_row.head()["Temp_level_low"])

#         processed_rows.append(df_row)

#         # Classifier check
#         high_volume_prob = clf.predict_proba(df_row)[:, 1][0]
#         high_volume = high_volume_prob >= 0.2  # e.g., 0.5

#         # Predict leachate if volume is high
#         pred = {}
#         pred["Measured"] = 1 if high_volume else 0
#         if high_volume:
#             for target in leachate_columns:
#                 model = regs[target]
#                 val = model.predict(df_row)[0]
                
#                 if target in log_targets:
#                     val = np.expm1(val)
#                 pred[target] = val
#         else:
#             for target in leachate_columns:
#                 pred[target] = 0  # or np.nan

#         # Store prediction
#         results.append(pred.copy())

#         # Update lag values for next timestep
#         for target in leachate_columns + ["Volume_leachate"]:
#             val = pred[target] if target in pred else 0
#             for lag in sorted(lag_settings, reverse=True):
#                 lag_values[f"{target}_lag{lag}"] = val if lag == 1 else lag_values[f"{target}_lag{lag-1}"]

#     return pd.DataFrame(results),  pd.concat(processed_rows, ignore_index=True)


# st.title("Rock Leachate Predictor")

# if "predict_clicked" not in st.session_state:
#     st.session_state["predict_clicked"] = False

# if st.button("Predict", key="predict_top", on_click=lambda: st.session_state.update({"predict_clicked": True})):
#     pass

# with st.expander("Input Rock Properties", expanded=True):
#     col_rock, col_events = st.columns(2)
#     with col_rock:
#         rock_input = {
#             "EC_rock": st.number_input("EC Rock", 2.3),
#             "Ph_rock": st.number_input("Ph Rock", 6.5),
#             "Corg_rock": st.number_input("Corg Rock", 1.2),
#             "SAR_rock": st.number_input("SAR Rock", 0.8),
#             "SiO2_rock": st.number_input("SiO2 Rock", 45.0),
#             "Al2O3_rock": st.number_input("Al2O3 Rock", 15.0),
#             "Fe2O3_rock": st.number_input("Fe2O3 Rock", 5.0),
#             "SO3_rock": st.number_input("SO3 Rock", 0.2),
#             "CaO_rock": st.number_input("CaO Rock", 8.0),
#             "MgO_rock": st.number_input("MgO Rock", 3.0),
#         }

#     st.divider()
#     with col_events:
#         num_events = st.slider("How many events?", 1, 10, 1)
#         events_list = []

#         for i in range(num_events):
#             st.subheader(f"Event {i+1}")
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 e_type = st.selectbox(f"Type {i+1}", ["rain", "snow"])
#             with col2:
#                 acid = st.number_input(f"Acid {i+1}", value=0)
#             with col3:
#                 temp = st.selectbox(f"Temp_level {i+1}", ["high", "low"])
            
#             events_list.append({
#                 "Type_event": e_type,
#                 "Acid": acid,
#                 "Temp_level": temp,
#             })

# if st.session_state.get("predict_clicked", False):
#     print("Button clicked")
    
#     raw_predictions, df_proc = predict_leachate(rock_input, events_list)

#     with st.expander("Results", expanded=True):
#         st.write("Predictions")
#         st.dataframe(raw_predictions)

#         st.divider()

#         X_shap = df_proc[feature_columns].fillna(-999)

#         explainer_clf = shap.TreeExplainer(clf, model_output='raw')
#         shap_values_clf = explainer_clf(X_shap)

#         for i in range(len(X_shap)):
#             st.subheader(f"Event {i+1}")
#             st.markdown("**Leaching Probability**")
#             prob = clf.predict_proba(X_shap.iloc[[i]])[0, 1]
#             st.metric("Probability", f"{prob:.2%}")
#             st.info(simple_explanation(shap_values_clf[i], "Chance of leaching"))  

#             fig, ax = plt.subplots(figsize=(6, 4))
#             shap.plots.waterfall(shap_values_clf[i], show=False, max_display=7)
#             st.pyplot(fig)
#             plt.close(fig)

#             if raw_predictions.iloc[i]["Measured"] == 1:
#                 st.markdown("#### Ion Concentrations")
#                 # Use tabs to organize the many ions
#                 ion_tabs = st.tabs(leachate_columns)
                
#                 for idx, ion in enumerate(leachate_columns):
#                     with ion_tabs[idx]:
#                         # Calculate SHAP for this specific event and ion
#                         reg_explainer = shap.TreeExplainer(regs[ion], model_output='raw')
#                         shap_values_reg = reg_explainer(X_shap.iloc[[i]])
                        
#                         fig_r, ax_r = plt.subplots(figsize=(6, 4))
#                         shap.plots.waterfall(shap_values_reg[0], show=False, max_display=7)
#                         st.pyplot(fig_r)
#                         plt.close(fig_r)

#                         val = raw_predictions.iloc[i][ion]
#                         st.metric(f"{ion} Value", f"{val:.2f}")

#                         st.caption(simple_explanation(shap_values_reg[0], f"{ion} level"))
#             else:
#                 st.caption("Predicted little to no leaching")

