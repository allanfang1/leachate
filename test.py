import joblib
from leachate_pipeline import LeachatePipeline 

pipeline = joblib.load("./pkl/pipeline.pkl")

rock_input = {
    'EC_rock': 1351, 'Ph_rock': 8.11, 'Corg_rock': 0.05, 'SAR_rock': 0.04,
    'SiO2_rock': 21.78, 'Al2O3_rock': 7.38, 'Fe2O3_rock': 3.16, 'SO3_rock': 24.65,
    'CaO_rock': 20.55, 'MgO_rock': 2.88
    }

events_list = [
{'Type_event':'rain', 'Acid':0, 'Temp_level':'low'},
{'Type_event':'rain', 'Acid':1, 'Temp_level':'low'},
{'Type_event':'rain', 'Acid':0, 'Temp_level':'high'},
]

print(pipeline.log_targets)
raw_predictions, df_proc = pipeline.predict_leachate(rock_input, events_list)
print(raw_predictions)