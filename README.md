# AFFIRM

Adaptable Forecasting Framework in RealtiMe (AFFIRM) is a framework that allows the user to easily create a machine learning model that can predict adverse events in the ICU before hours before they occur. Predicting adverse events before they occur can help clinicians make decisions on how to treat the patient. The framework is specialised for predicting atrial fibrillation however it can be used to predict any adverse event. 


## Why use AFFIRM
It is useful as a baseline prediction for event prediction in the ICU research.
It is quick and easy to use.
Easily adapatable


## Data

AFFIRM uses the HiRID dataset that was created by Hyland et al. https://physionet.org/content/hirid/ To use AFFIRM you will first need to get permission to use HiRID. 

## Getting started
Download the *raw\_stage* data from HiRID after getting permission and place it in *data\_path*. Also place the files from *extra\_files* from this repository into the *raw\_stage* folder.
      
    
    ./data_path/
	├── raw_stage          
	    ├── observation_tables       
	        ├── part-0.parquet         
	        ├── ...         
	        ├── part-250.parquet         
	    ├── pharma_records         
	        ├── part-0.parquet     
	        ├── ...         
	        ├── part-250.parquet        
	    ├── APACHE_groups.csv
	    ├── drug_classes.csv 
	    ├── general_table.csv
	    ├── hirid_variable_reference.csv


## Index

   * [AFFIRM](#affirm)
   	* [Preprocess](#preprocess)
   	* [Prepare](#prepare)
   	* [Predict](#predict)

## Preprocess
```python
preprocess_params = {
    'rename_dict' : {'temp':'Temperature','mean.arterial.pressure':'MAP','systolic.arterial.pressure':'Systolic BP',
               'diastolic.arterial.pressure':'Diastolic BP'},
    'parameter_dict' : {'Circadian_rhythm': [4, 10]},
     
    'filter_range': [0.01, 0.99]
}
affirm.fit_preprocess(**preprocess_params)
affirm.preprocess()
```python
```
## Prepare
```python
prepare_params = {
    'predict_hours': 6,                 
    'grouping_hours': 1,
    'group_how_list': ['max'],#,'min'],
    'group_label_within':120, 
    'rolling': False,
    'take_first': False,
    'percentage_patients_per_variable': 0.8, 
    'avg_values_each': 2,
    'feature_names': [],
    'pharma_quantile' : 0.75,
    'include_patients':[],
    'exclude_patients': [],#'Surgical Cardiovascular'
    
}
affirm.fit_prepare(**prepare_params)
# affirm.prepare()
```python
```

## Predict
```python
def create_keras():  
    model = Sequential()
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[AUC(curve="PR", name='auprc')])        
    return model

keras_model = create_keras()
keras_model = KerasClassifier(build_fn= create_keras, verbose=0)
keras_model._estimator_type = "classifier"
predict_params = {
    'models': {
#         'Logistic Regression': LogisticRegression(random_state=0)
#           , 'Keras':keras_model,
#           'Random Forest':RandomForestClassifier(max_depth=4, random_state=0),
#             'LightGBM':lgb.LGBMClassifier(boosting_type='gbdt', objective='binary'),
#         'Optimised XGBoost': xgb.XGBClassifier(base_score=0.5, 
#                                                   colsample_bylevel=1, 
#                                                   colsample_bytree=1,
#                                                   gamma=0, 
#                                                   learning_rate=0.1, 
#                                                   max_delta_step=0, 
#                                                   max_depth=10,
#                                                   min_child_weight=1, 
#                                                   n_estimators=100, 
#                                                   nthread=-1,
#                                                   objective='binary:logistic', 
#                                                   reg_alpha=0, 
#                                                   reg_lambda=1,
#                                                   scale_pos_weight=1, 
#                                                   eval_metric = 'aucpr',
#                                                   seed=0, 
#                                                   missing = 1,
#                                                   subsample=1,
#                                                   use_label_encoder=False), 
        'XGBoost': xgb.XGBClassifier(objective = "binary:logistic",
           eval_metric = "aucpr",use_label_encoder=False)
         },
    'colors' : {'LightGBM': '#4e8542',# dark green
          'Baseline': '#ff9292', #pink
          'Logistic Regression':'#eccd1c', #gold
          'Random Forest': '#6aa4c8', #sky blye
          'XGBoost': '#ff833c', #organ
            'Optimised XGBoost': '#fcaf83',
          'Keras': '#8dd8d3' #light blue
         },
    'n_splits': 2,
    'keep_top_features': 20,
    'intervention_features': ['Potassium', 'Magnesium']
     
}
affirm.fit_predict(**predict_params)
```python
```



![alt text](https://github.com/Ti-Yao/AFFIRM/blob/main/AFFIRM.png?raw=true)
