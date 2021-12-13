# import modules
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import lightgbm as lgb
import xgboost as xgb
import sys
sys.path.append('../')
from utils import *
from datetime import datetime
from scipy.signal import find_peaks, detrend, savgol_filter
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")

# import sklearn modules
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, plot_roc_curve,roc_auc_score, precision_recall_curve,average_precision_score,f1_score,auc,confusion_matrix,plot_precision_recall_curve,plot_confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

# import tensorflow modules
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path

Nfiles = 250
working_directory = ''

# path to the data **CHANGE path if necessary**
# data_path = '/media/tina/Seagate Expansion Drive/Work/hirid/'
class AFFIRM:
    def __init__(self, data_path, label_name):
        self.data_path = data_path
        self.raw_path = f'{data_path}/raw_stage/'
#         self.check_data_path(data_path)
        self.preprocessed_path = f'{data_path}/preprocessed_stage/{label_name}/'
        self.label_name = label_name

#     def check_data_path(self, data_path):
#         Path(data_path + 'prepared_stage/' + label_name).mkdir(parents=True, exist_ok=True)
#         Path(data_path + 'preprocessed_stage/' + label_name).mkdir(parents=True, exist_ok=True)


    def fit_preprocess(self, rename_dict, parameter_dict, filter_range, label_data = pd.Series()):
        self.rename_dict = rename_dict
        self.parameter_dict = parameter_dict
        self.label_data = label_data
        self.filter_range = filter_range

    def preprocess(self):
        static = []
        count = 0
#         Nfiles = 250
        for Nfile in tqdm(range(Nfiles)):
#             printProgressBar(Nfile + 1, Nfiles)
            raw_df = pd.read_parquet(self.raw_path + f'observation_tables/parquet/part-{Nfile}.parquet'
                                         ,columns=['datetime','patientid','value','variableid'])
            pharma_df = pd.read_parquet(self.raw_path + f'pharma_records/parquet/part-{Nfile}.parquet'
                                ,columns=['givenat','patientid','givendose','pharmaid'])
            df, static_df, pharma_df = self.preprocess_pipeline(raw_df, pharma_df)
            save_data(df, self.preprocessed_path + f'observation_tables/part-{Nfile}.parquet')
            save_data(pharma_df, self.preprocessed_path + f'pharma_records/part-{Nfile}.parquet')

            static.append(static_df)
#             count += get_Ncases(df, 'AF')
        static_df = pd.concat(static)
        general = pd.read_csv(self.raw_path + 'general_table.csv', usecols = ['patientid']) # from hirid
        static_df['patient'] = static_df.reset_index().patientid.values
        static_df = general.merge(static_df, on = 'patientid', how = 'left').set_index('patientid')
        save_data(static_df, self.preprocessed_path + 'static_df.parquet')

    def preprocess_pipeline(self,raw_df, pharma_df):
        # preprocess static data, change from long form to wide form and change height and weight to bmi
        static_df = self.preprocess_static(raw_df)

        # preprocess pharmacological data and categorises the variables by drug class, changes the datetime to minutes since admission, every 5 minutes, and changes values to a binary flag
        pharma_df = self.preprocess_pharma(pharma_df)

        # preprocess observation data, changes the datetime to minutes since admission, every 5 minutes
        df = self.preprocess_observations(raw_df)

        # groups variables according to rename_dict i.e. axillary and core temperature are just labelled as temperature. The mean is taken if there is more than one value every 5 minutes.
        df = self.rename_variables(df)

        # change from long form data to wide form: patientid, minutes_since_admission, variable1, variable2...
        df = self.pivot(df)
        pharma_df = self.pivot(pharma_df)

        # labels datapoints according to the parameter dict. Labelled data will be added if specified and therefore don't need the parameter dict in this case.
        df = self.define_labels(df)

        # filters out the values between the filter range as a percentile, this is to remove anomalous values
        df = self.filter_percentile(df)


        return df, static_df, pharma_df

    def id_to_name(self, df, source):
        '''
        Converts HiRID variable IDs to variable names using the reference table.
        Inputs:
        df - the dataframe that has variable ids
        source - observation and pharma ids are different in HiRID so need to be specified
        Outputs:
        df - dataframe that has variable names
        '''
        reference = pd.read_csv(self.raw_path  + 'hirid_variable_reference.csv') # from hirid
        if source == 'observations':
            reference = reference.loc[reference['Source Table'] == 'Observation'] # selects the observation variable ids
        elif source == 'pharma':
            reference = reference.loc[reference['Source Table'] == 'Pharma'] # selects the pharmacological variable ids
        reference = reference[['ID','Variable Name']]
        reference = reference.rename(columns = {'Variable Name':'variable_name'})
        df = df.merge(reference[['ID','variable_name']], left_on = 'variableid', right_on = 'ID') # replace variableid with variable_name
        return df

    def time_to_mins_since_adm(self, df):
        '''
        Changes the datetime to minutes since admission.
        Inputs:
        df - dataframe with datetime
        parameter_dict - the dictionary that defines the labelling
        Outputs:
        df - dataframe with minutes since admission
        '''
        general = pd.read_csv(self.raw_path + 'general_table.csv', parse_dates = ['admissiontime']) # from hirid
        df = df.merge(general, on = 'patientid') # add static information
        df['mins_since_adm'] = (df.datetime-df.admissiontime) # change datetime to time since admission
        df = df[['patientid','mins_since_adm','value','variable_name']]
#         if 'Circadian_rhythm' in self.parameter_dict:
#             if isinstance(self.parameter_dict['Circadian_rhythm'], list):
#                 df = df[~((df['variable_name'] == 'Circadian_rhythm') & (df['value'].isin(self.parameter_dict['Circadian_rhythm'])))] # removes circadian rhythm as a variable if it is part of the label because e.g. circadian rhythm = 10 is equivalent to AF
#             else:
#                 df = df[~((df['variable_name'] == 'Circadian_rhythm') & (df['value'] != self.parameter_dict['Circadian_rhythm']))] # removes circadian rhythm as a variable if it is part of the label because e.g. circadian rhythm = 10 is equivalent to AF
        df = df.groupby(['patientid','variable_name',pd.Grouper(key = 'mins_since_adm',freq = '5min')]).max() # group time since admission every 5 mins
        df = df.reset_index()
        df.mins_since_adm = (df.mins_since_adm/ np.timedelta64(1, 'm')).astype(int) # change mins since admission to integer
        df.mins_since_adm = df.mins_since_adm.apply(lambda x: int(x/5) * 5)
        df = df.set_index(['patientid','mins_since_adm']).sort_index() # set index
        return df

    def pivot(self, df):
        '''
        For changing the long form data (patientid, minutes since admission, variable_names, values ) to wide form data (patientid, minutes since admission, variable_name_1, variable_name_2, ..., variable_name_N) where patientid and minutes since admission make the index
        '''
        df = df.pivot_table('value', ['patientid','mins_since_adm'], 'variable_name') # pivto table
#         df = df.groupby('patientid').apply(lambda x: fill_timepoints(x))
#         df['time_since_admission'] = df.reset_index()['mins_since_adm'].values
        df = df.sort_index(axis=1)
        return df

    def preprocess_static(self, df):
        '''
        For preprocessing static data. Outputs a dataframe with the age, sex and BMI for each patient
        '''
#         printProgress(sys._getframe(  ).f_code.co_name)
        general = pd.read_csv(self.raw_path + 'general_table.csv', parse_dates = ['admissiontime']) # from hirid
        if 'variableid' in df.columns:
            df = self.id_to_name(df, 'observations') # changes variable id to variable name
        height = df.loc[df.variable_name.str.contains('height', case = False)].variable_name.unique()[0] # height variable
        weight = df.loc[df.variable_name.str.contains('weight', case = False)].variable_name.unique()[0] # weight variable

        static_df = df.loc[(df['variable_name'] == weight) | (df['variable_name'] == height)] # gets weights and heights of patients
        static_df = pd.pivot_table(static_df, values = 'value', index = 'patientid', columns= 'variable_name') # pivot table
        static_df = static_df.merge(general, on = 'patientid') # add static information
        static_df.sex =  np.where(static_df['sex']== 'M', 1, 0) # change sex from character to integer 1 for male and 0 for female
        static_df = static_df.set_index('patientid') # make patient id index
        static_df['bmi'] = static_df[weight].div((static_df[height]/100)**2) # calculate bmi from height and weight
        static_df = static_df[['sex','age','bmi']] # chooose only sex age and bmi
        return static_df


    def preprocess_observations(self, df):
        '''
        preprocessing the hirid observation data
        '''
#         printProgress(sys._getframe(  ).f_code.co_name)
        if 'variableid' in df.columns:
            df = self.id_to_name(df, 'observations') # change variableids to variable names
        # cleans the variable names
        df.variable_name = df.variable_name.str.replace(' ', '_')
        df.variable_name = df.variable_name.str.replace('[^A-Za-z0-9_]+', '', regex=True)
        df = self.time_to_mins_since_adm(df) # changes datetime to minutes since admission
        df.loc[(df['value'] == 0) & ~df['variable_name'].str.contains('scale'), 'value'] = np.nan # removes values that are 0 unless they are a part of a scale e.g. the glasgow coma scale.
        df = df.dropna()
        return df

    def preprocess_pharma(self, df):
        '''
        Preprocess pharmacological data from HiRID.
        '''
#         printProgress(sys._getframe(  ).f_code.co_name)
        drug_classes = pd.read_csv(self.raw_path + 'drug_classes.csv') # made from researching the different drug names
        df = df.rename(columns = {'pharmaid':'variableid','givenat':'datetime','givendose':'value'}, errors='ignore') # rename
        df = self.id_to_name(df, 'pharma') # convert variableid to variable name
        df = df.merge(drug_classes, on = 'variable_name') # label variable names with drug classes
        df.loc[df[self.label_name] ==1,'drug_class'] = 'Intervention'
        df = df[['patientid','datetime','drug_class','value']] # keep the drug classes not the individual pharma variables
        df = df.rename(columns = {'drug_class':'variable_name'}) # change drug classes to variable name
        df = self.time_to_mins_since_adm(df) # change datetime to minutes since admission
        df.value = 1 # add a binary presence flag.
#         df = pd.pivot_table(df, values = 'value', index = ['patientid','mins_since_adm'], columns= 'variable_name')

        return df



    def define_labels(self,df):
        '''
        Creates the labelled data, if no labelled data is specified, then adds the labels to the observational data.
        '''
#         printProgress(sys._getframe(  ).f_code.co_name)
        if self.label_data.empty: # if labelled data is not specfied
            label_data = pd.Series(dtype = np.float64, index = df.index).fillna(True) # initialise a series for labels
            for variable, value in self.parameter_dict.items(): # for each parameter specified
                if isinstance(value, dict):
                    min_val, max_val = value['min'], value['max']
                    label_data = label_data & (df[variable] >= min_val) & (df[variable] <= max_val) # label data that is
                elif isinstance(value, list): # if the values of the parameter are a list
                    label_data = label_data & (df[variable].isin(value))
                else:
                    label_data = label_data & (df[variable] == value) # if the value is an integer then just take the integer value

        label_data.name = self.label_name # name the labels
        df = df.merge(label_data, on = ['patientid','mins_since_adm'], how = 'left') # merge the labels to the observational data
        df = df.groupby('patientid').apply(lambda x: self.fill_timepoints(x)) # regularises the time interval
        df[self.label_name] = df[self.label_name].fillna(0) * 1 # make all the labels 0 where they are not 1
        df[self.label_name] = df[self.label_name].astype('int64') # change the labels to integer
        df = df.sort_index(axis=1)
        df = df[[self.label_name] + [ col for col in df.columns if col != self.label_name]] # sorts the column names
        return df

    def fill_timepoints(self, df):
        '''
        Previously data is changed from datetime to minutes since admission (rounded to 5). This results in irregular time points e.g. 5, 25, 100, 120,..., N. filling the timepoints changes this to e.g. 0, 5, 10, 15,...,N
        '''
        df = df.loc[df.index[0][0]] # patient by patient
        new_df = pd.DataFrame() # new dataframe
        new_df['mins_since_adm'] = np.arange(0,df.index.max() + 5,5) # resamples the timepoints
        new_df = new_df.merge(df, on = 'mins_since_adm',how = 'left') # adds data to new dataframe
        new_df = new_df.set_index('mins_since_adm')
        return new_df

    def rename_variables(self, df):
        '''
        renames and groups variable names that measure the same thing. E.g. axillary temperature and core temperature can be renamed to temperature.
        '''
#         printProgress(sys._getframe(  ).f_code.co_name)
        for regex, name in self.rename_dict.items():
            df.loc[df['variable_name'].str.contains(regex) == True, 'variable_name'] = name
        return df



    def filter_percentile(self, df):
        '''
        Filters out anomalous values by taking a range of values between two percentiles. Filters each variable until the standard deviation is less than three times the mean.
        '''
#         printProgress(sys._getframe(  ).f_code.co_name)
        min_percentile = min(self.filter_range) # minimum percentile
        max_percentile = max(self.filter_range) # maximum percentile
        skipfilter = ['code', 'mode', 'score', 'count', 'group']
        filter_var = list(df.drop(columns = [self.label_name,'time_since_admission'], errors = 'ignore').columns.values) # variables to filter
        filter_var = [var for var in filter_var if not any(skip in var for skip in skipfilter)] # variables to filter
        df = df[df.index.get_level_values(1) >= 0] # filter out values that are measured before admission
        for variable in filter_var:
            min_val, max_val = df[variable].quantile(min_percentile), df[variable].quantile(max_percentile) # get the minimum and maximum range of the values for the given variable
            df.loc[~df[variable].between(min_val, max_val),variable] =np.nan # remove the values outside the range
            if df[variable].std() > df[variable].mean() * 3: # while the standard deviation of the variables is greater than three times the mean
                min_val, max_val = df[variable].quantile(0.1), df[variable].quantile(0.9) # get the minimum and maximum range of the values for the given variable
                df.loc[~df[variable].between(min_val, max_val),variable] =np.nan # remove the values outside the range
        return df





    '''
    Pipeline for preparing data for being classified, data preparation is based on the inputs:
    • df - dataframe containing observation data and labels
    • pharma_df - dataframe containing pharmacological data
    • predict_hours - prediction window
    • label_name - the name of the adverse event
    • grouping_hours - number of hours to group data.
    • group_how - how to group the data within grouping hours, 'max', 'min', 'mean' or 'median'
    • group_label_within - label all timepoints as positive if they are within group_label_within minutes of a positive time point
    • rolling - choose whether it will be a rolling prediction every 5 minutes, this may take a lot more computation time
    • take_first - Boolean. Whether or not to only take the first adverse event for every patient and ignore the others
    '''
    def fit_prepare(self, feature_names, percentage_patients_per_variable, avg_values_each, pharma_quantile,include_patients, exclude_patients, predict_hours, grouping_hours, group_how_list, group_label_within, rolling, take_first):
        if not pd.Series(group_how_list).isin(pd.Series(['max', 'min', 'median', 'mean'])).all():
            raise ValueError('change group_how_list')
        self.predict_minutes = int(predict_hours * 60)
        self.predict_hours = predict_hours
        self.grouping_hours = grouping_hours
        self.grouping_minutes = int(grouping_hours * 60)
        self.group_label_within = group_label_within
        self.rolling = rolling
        self.take_first = take_first
        self.group_how_list = group_how_list
        self.percentage_patients_per_variable = percentage_patients_per_variable
        self.avg_values_each = avg_values_each
        self.include_patients = include_patients
        self.exclude_patients = exclude_patients
        self.prepared_path = self.get_prepared_path()
        self.pharma_quantile = pharma_quantile
        self.feature_names = feature_names
        self.results_path = working_directory + f'results/{self.label_name}/{self.prepared_path.split(self.label_name)[-1]}/'

    def get_prepared_path(self):
        if len(self.include_patients) != 0 and len(self.exclude_patients) != 0:
            raise ValueError('Only specify include_patients or exclude_patients, not both!')
        prepared_path = f'{self.data_path}/prepared_stage/{self.label_name}/'
        prepared_path += f'rolling={self.rolling}/'
        prepared_path += f'take_first={self.take_first}/'
        if len(self.include_patients) == 0 and len(self.exclude_patients) == 0:
            prepared_path += 'all/'
        elif len(self.include_patients) != 0:
            prepared_path += f'include/{self.create_patient_group_label(self.include_patients)}/'
        elif len(self.exclude_patients) != 0:
            prepared_path += f'exclude/{self.create_patient_group_label(self.exclude_patients)}/'
        prepared_path += f'grouping={self.grouping_hours}/predict={self.predict_hours}/'
        return prepared_path

    def prepare(self):
#         Nfiles = 250
        for group_how in self.group_how_list:
            obs = []
            pharma = []
            group_how = group_how.lower()
            for Nfile in tqdm(range(Nfiles)):
#                 printProgressBar(Nfile + 1, Nfiles)
                df = pd.read_parquet(self.preprocessed_path + f'observation_tables/part-{Nfile}.parquet')
                pharma_df = pd.read_parquet(self.preprocessed_path + f'pharma_records/part-{Nfile}.parquet')
                df, pharma_df = self.prepare_pipeline(df, pharma_df, group_how)
                obs.append(df)
                pharma.append(pharma_df)

            df = pd.concat(obs)
            pharma_df = pd.concat(pharma)
            feature_names = list(df.drop(columns = [self.label_name,'time_since_admission'], errors = 'ignore').columns.values)
            df = df[feature_names].add_suffix('_' + group_how).merge(df[[self.label_name]], on = df.index.names)
            save_data(df, self.prepared_path + f'df_{group_how}.parquet')
            save_data(pharma_df, self.prepared_path + f'pharma_df.parquet')




    def prepare_pipeline(self, df, pharma_df, group_how):
        if len(self.exclude_patients) != 0 or len(self.include_patients) != 0:
            df = self.include_exclude_patients(df)
        df, pharma_df = self.variable_chooser(df, pharma_df)
        df = self.prepare_labels(df)
        if self.rolling == True:
            df = self.group_rolling(df, group_how)
            pharma_df = self.group_rolling(pharma_df, 'max')
        else:
            df = self.group_hours(df, group_how)
            pharma_df = self.group_hours(pharma_df, 'max')
        return df, pharma_df

    def prepare_labels(self, df):
#         printProgress(sys._getframe(  ).f_code.co_name)
        df = df.groupby('patientid').apply(lambda x: self.group_label(x)) # label all timepoints positive if they are within group_label_within
#         df = self.filter_patients(df) # filter out some patients
        if 'AF' in self.label_name:
            df = df.groupby('patientid').apply(lambda x: self.annotate_AF(x)) # if predicting AF, annotate some extra timepoints
        if self.take_first == True: # take the first adverse event, true or false
            df = df.groupby('patientid').apply(lambda x: self.take_first_case(x))
        df = df.groupby('patientid').apply(lambda x: self.shift_label(x, self.label_name)) # shift labels back by the prediction window and label all timepoints from that point to the current positive point.
        return df


    def variable_chooser(self, df, pharma_df):
        '''
        Reduces the numnber of variables depending on some inputs.
        Inputs:
        • feature_names - the user can choose their own list of variables to keep
        alternatively
        • percentage_patients_per_variable - the percentage of patients that have a certain variable
        • avg_values_each - average number of measurements per patient
        The default values are 0.8 and 2 respectively. This means the variables of which 80% of patients have on average 2 measurements for are kept. To keep all the values choose 1 and 0 respectively
        '''
#         printProgress(sys._getframe(  ).f_code.co_name)
        if self.percentage_patients_per_variable > 1 or self.percentage_patients_per_variable < 0:
            raise ValueError('percentage_patients_per variable must be between 0 and 1!')
        if len(self.feature_names) == 0:
            obs_feature_names = self.reduce_obs_var(df)
            pharma_feature_names = self.reduce_pharma_var(pharma_df)
            feature_names = obs_feature_names + pharma_feature_names + [self.label_name]
        else:
            feature_names = self.feature_names + [self.label_name]
        if 'Circadian_rhythm' in self.parameter_dict and 'Circadian_rhythm' in feature_names:
            feature_names.remove('Circadian_rhythm') # remove circadian rhythm if AF in feature names
        feature_names = list(set([x.replace('_max', '').replace('_min', '').replace('_mean','').replace('_median','') for x in feature_names]))
        df = df[df.columns.intersection(feature_names)] # keep only columns listed in feature names
        pharma_df = pharma_df[pharma_df.columns.intersection(feature_names)] # keep only columns listed in feature names
        return df, pharma_df

    def reduce_obs_var(self, df):
        N_patients = len(df.reset_index().patientid.unique())
        min_patients_per_variable = self.percentage_patients_per_variable * N_patients
        N_patients_per_variable = df.groupby('patientid').count().replace(0, np.nan).count()
        feature_names1 = N_patients_per_variable[N_patients_per_variable >  min_patients_per_variable].index.values
        var_counts = df.describe().iloc[0]
        feature_names2 = var_counts[var_counts > N_patients * self.avg_values_each].index.values
        feature_names = sorted(list(set(feature_names1).intersection(set(feature_names2))))
        return feature_names

    def reduce_pharma_var(self, pharma_df):
        pharma_var_counts = pharma_df.describe().iloc[0] # measurements per pharmacological variable
        feature_names = list(pharma_var_counts [pharma_var_counts > pharma_var_counts .quantile(self.pharma_quantile)].index.values) # reduce number of pharmacological variables depending on the quantile chosen
        return feature_names

    def include_exclude_patients(self, df):
        '''
        _ or exclude patient cohorts depending on their APACHE labels from HiRID. e.g. cardiac, surgical cardiac, trauma, sepsis...
        '''
#         printProgress(sys._getframe(  ).f_code.co_name)
        apache_ref = pd.read_csv(self.raw_path + 'APACHE_groups.csv').set_index('Name') # from hirid

        apache_df = df.reset_index().set_index('patientid')[['APACHE_II_Group','APACHE_IV_Group']].dropna(thresh = 1)
        all_pids = apache_df.index.tolist()
        # exclude patients
        if len(self.exclude_patients) != 0 and len(self.include_patients) == 0:
            exclude_pids = []
            for patient_type in self.exclude_patients:
                ii_val, iv_val = apache_ref.loc[patient_type]
                exclude_apache_df = apache_df[(apache_df['APACHE_II_Group'] == ii_val) | (apache_df['APACHE_IV_Group'] == iv_val)]
    #             print(exclude_apache_df)
                exclude_pids += exclude_apache_df.index.tolist()
    #         print(exclude_pids)
            pids = list(set(all_pids).difference(set(exclude_pids)))

        # include patients
        if len(self.include_patients) != 0 and len(self.exclude_patients) == 0:
            pids = []
            for patient_type in self.include_patients:
                ii_val, iv_val = apache_ref.loc[patient_type]
                include_apache_df = apache_df[(apache_df['APACHE_II_Group'] == ii_val) | (apache_df['APACHE_IV_Group'] == iv_val)]
                pids += include_apache_df.index.tolist()
            pids = list(set(pids))
        df = df.loc[list(set(pids))]
        return df

    def create_patient_group_label(self, patients):
        patient_groups = [re.sub('Surgical[ \_]','', word) for word in patients]
        group_label = []
        for group in set(patient_groups):
            if patient_groups.count(group) > 1:
                group_label.append(group+ ('(All)'))
            else:
                group_label.append([s for s in patients if group in s][0])
        group_label = '_'.join(group_label)
        return group_label


    def group_rolling(self, df, group_how):
        '''
        Takes the max, min, mean or median within "grouping hours" every five minutes
        '''
#         printProgress(sys._getframe(  ).f_code.co_name)
        feature_names = list(df.drop(columns = [self.label_name,'time_since_admission'], errors = 'ignore').columns.values)
        grouping_index = int(self.grouping_minutes/5)
        group_df = df.groupby('patientid')
        if group_how == 'max':
            df[feature_names] = group_df[feature_names].apply(lambda x: x.rolling(grouping_index).max())
        elif group_how == 'min':
            df[feature_names] = group_df[feature_names].apply(lambda x: x.rolling(grouping_index).min())
        elif group_how == 'mean':
            df[feature_names] = group_df[feature_names].apply(lambda x: x.rolling(grouping_index).mean())
        elif group_how == 'median':
            df[feature_names] = group_df[feature_names].apply(lambda x: x.rolling(grouping_index).median())
        if self.label_name in df.columns:
            df[[self.label_name]] = group_df[[self.label_name]].apply(lambda x: x.rolling(grouping_index).max())
            df[self.label_name] = df[self.label_name].dropna(how = 'all')
            df = df.groupby('patientid').apply(lambda df: df.loc[df.index[0][0]].loc[self.grouping_minutes:])
        df = self.reduce_variables(df, group_how)
        return df

    def reduce_variables(self, df, group_how):
        feature_imp_path = self.results_path.replace('rolling=True','rolling=False')
        reduced_features = pd.read_csv(feature_imp_path + 'feature_importances.csv', index_col= 0).head(20).index.tolist()
        reduced_features = [s for s in reduced_features if f"_{group_how}" in s]
        reduced_features = list(set([x.replace(f"_{group_how}", '') for x in reduced_features]))
        if self.label_name in df.columns:
            reduced_features += [self.label_name]
        df = df[list(set(reduced_features).intersection(set(df.columns)))]
        return df

    def group_hours(self, df, group_how):
        '''
        Takes the max, min, mean or median within "grouping hours"
        '''
#         printProgress(sys._getframe(  ).f_code.co_name)
        feature_names = list(df.drop(columns = [self.label_name,'time_since_admission'], errors = 'ignore').columns.values)
        new_df = pd.DataFrame()
        df = df.reset_index()
        df = df.groupby(['patientid', df.mins_since_adm//self.grouping_minutes])
        if group_how == 'max':
            new_df[feature_names] = df[feature_names].max()
        elif group_how == 'min':
            new_df[feature_names] = df[feature_names].min()
        elif group_how == 'mean':
            new_df[feature_names] = df[feature_names].mean()
        elif group_how == 'median':
            new_df[feature_names] = df[feature_names].median()
        try:
            new_df[[self.label_name]] = df[[self.label_name]].max()
        except:
            pass
        new_df = new_df.reset_index()
        new_df = new_df.rename(columns = {'mins_since_adm':'hours_since_adm'})
        new_df.hours_since_adm = new_df.hours_since_adm *self.grouping_hours
        new_df = new_df.set_index(['patientid','hours_since_adm'])
        df = new_df
        return df



    def group_label(self, df):
        '''
        Label all timepoints positive if they are within group_label_within
        '''
        df = df.loc[df.index[0][0]] # patient by patient
        if df[self.label_name].max() == 1:
            group_times = {}
            i = 0
            case_times = list(df.loc[df[self.label_name] == 1].index)
            group_times[i] = []
            if len(case_times) == 1:
                return df
            else:
                for j in range(1, len(case_times)):

                    if (case_times[j] - case_times[j-1]) < self.group_label_within:
                        group_times[i] += list(np.arange(case_times[j-1],case_times[j]+5,5))
                    else:
                        i+=1
                        group_times[i] = []

                group_times = [k for value in group_times.values() for k in value ]
                df.loc[df.index.isin(group_times),self.label_name] = 1
                if len(group_times) == 0:
                    return None
                else:
                    return df
        else:
            return df



    def filter_patients(self, df):
        '''
        remove patients who have a label within grouping hours since admission or if they have a length of stay less than grouping hours
        '''
        min_case_time = self.grouping_minutes
        los = df.reset_index().groupby('patientid').mins_since_adm.max() # length of stay of patients
        all_patients = los[los > min_case_time].index.values # patients who have length of stay greater than the grouping hours

        max_case  = df.loc[df.loc[df[self.label_name] == 1].reset_index().patientid.unique()].groupby('patientid').apply(lambda x: np.where(x[self.label_name] == x[self.label_name].max())[-1][-1] * 5) # the latest case
        remove_case_patients = max_case[max_case < min_case_time].index.values # if the latest case is less than the grouping hours remove the patients

        keep_patients = set(all_patients).difference(set(remove_case_patients))
        df = df.loc[keep_patients]

        return df



    def shift_label(self, df, label_name):
        '''
        Shift the positive labels of each patient back by the prediction window and fill all the time points in between so that the labels represent whether the patient will develop the adverse event within the time window. I.e. the user may be trying to see if the patient will develop the event in the next 6 hours, then the 5 minutes before the event will still be a positive label because 5 minutes is still within 6 hours.
        '''

        df = df.loc[df.index[0][0]] # per patient
        case_times = pd.Series(df.index[df[label_name] == 1]) # positive label times
        case_times = case_times.apply(lambda x: list(np.arange(x-self.predict_minutes, x, self.grouping_minutes))).tolist() # make all the time points in the prediction window positve
        case_times = sorted(list(set([item for sublist in case_times for item in sublist]))) # take the unique time points
        df.loc[df.index.isin(case_times),label_name] = 1 # label them posirv
        return df

    def take_first_case(self, df):
        df = df.loc[df.index[0][0]]
        padding_hours = self.predict_minutes# + self.grouping_hours
        if df[self.label_name].max() ==1:
            first_index = df[self.label_name].eq(1).idxmax()
            df = df.loc[:first_index+padding_hours]
        if df.empty:
            return None
        else:
            return df


    def annotate_AF(self, df):
        '''
        Adding AF annotations based on whether there is an elevated or a peak in heart rate.
        '''

        df = df.loc[df.index[0][0]]
        if df[self.label_name].max() == 1:
            time = df.index
            heart_rate = df['Heart_rate']
            case_times = list(df.loc[df[self.label_name] == 1].index)
            if len(case_times) == 0:
                return None
            else:
                over_100 = pd.Series(df.loc[df['Heart_rate'] > 100].index.values)
                over_100 = over_100.loc[(over_100 > min(case_times) - self.group_label_within) & (over_100 < max(case_times) + self.group_label_within)]
                yhat = savgol_filter(heart_rate, 11, 4)
                yhat = detrend(yhat, type ='constant')
                height = pd.DataFrame(yhat, columns = ['heart_rate'])
                height = height.loc[height['heart_rate'] > 0]
                height['z'] = height.heart_rate.apply(lambda x: (x - height.heart_rate.mean())/height.heart_rate.std())
                height = height.loc[height['z'] > 2].heart_rate.min()

                peaks, properties = find_peaks(yhat, height = height)
                properties['peaks'] = peaks * 5
                properties = pd.DataFrame(properties)
                peaks = properties.peaks.loc[(properties.peaks > min(case_times) - self.group_label_within) & (properties.peaks < max(case_times) + self.group_label_within)]
                peaks = list(set(list(peaks.values) + list(over_100) + case_times))
                peaks = sorted(peaks)
                group_times = {}
                interval = 180
                i = 0
                group_times[i] = []
                if len(peaks) == 1:
                    group_times[0] = peaks
                else:
                    for j in range(1, len(peaks)):
                        if (peaks[j] - peaks[j-1]) < self.group_label_within:
                            group_times[i] += list(np.arange(peaks[j-1],peaks[j]+5,5))
                        else:
                            i+=1
                            group_times[i] = [peaks[j-1]]
                new_case_times = []
                for key in group_times.keys():
                    new_case_times += group_times[key]
                df.loc[df.index.isin(new_case_times), self.label_name] = 1
                return df
        else:
            return df

    def fit_predict(self, models, colors, n_splits, keep_top_features = None, base_features = [], intervention_features = []):
        self.models = models # list of machine learning models to compare
        self.colors = colors
        self.model_shap_values = {} # shap values for each model
        self.cv =  StratifiedKFold(n_splits=n_splits, random_state = 42, shuffle = True) # cross validation
        self.mean_vals = np.linspace(0, 1, 100)
        self.base_features = base_features
        self.intervention_features = intervention_features
        self.df = self.combine_df()
        self.X, self.X_base, self.y = self.get_X_y(self.df)
        self.keep_top_features = keep_top_features
        self.results = {}

    def plot_cm(self, model_name, model,X_test, y_test ):
        '''
        plot and save the confusion matrix for each model
        '''

        y_pred = model.predict(X_test)
        y_pred = (y_pred >0.5)  * 1
        f1 = f1_score(y_test,y_pred)
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_confusion_matrix(model, X_test, y_test, cmap = 'Blues', ax = ax)
        ax.set(title = f'{model_name}. F1 Score = {f1:.2f}')
        save_data(fig, self.results_path + f'{model_name}/confusion_matrix.png')

    def get_top_features(self, model_auprc):
        '''
        Turns shap values to feature importance
        '''
        model_auprc =  {k: v for k, v in sorted(model_auprc.items(), reverse = True, key=lambda item: item[1])}
        for model_name in model_auprc:
            try:
                shap_values = self.model_shap_values[model_name]
                break
            except:
                pass
        # model_name = max(model_auprc, key= model_auprc.get)
        # shap_values = self.model_shap_values[model_name]
        feature_importance = pd.DataFrame(np.abs(shap_values.values).mean(0),index=self.feature_names, columns = ['feature_importance'])
        feature_importance = feature_importance.sort_values(by = 'feature_importance', ascending = False)
        save_data(feature_importance, self.results_path + f'feature_importances.csv')
        top_features = feature_importance.head(self.keep_top_features).index.values
        return top_features


    def initial_prediction(self):
#         X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify = self.y)
        train_df, test_df = train_test_split(self.df, test_size = 0.2, random_state = 42, stratify = self.df[self.label_name].values)
        X_train, _ , y_train= self.get_X_y(train_df)
        X_test, _ , y_test= self.get_X_y(test_df)
        scaler = MinMaxScaler(feature_range=(-1, 1))#StandardScaler() # scale the data
        X_train  = scaler.fit_transform(X_train) # scale the training data
        X_test = scaler.transform(X_test) # scale the testing data
        X_train = self.interpolate_values(X_train) # interpolate or impute values
        X_test = self.interpolate_values(X_test) # interpolate or impute values

        test_df = test_df.groupby('patientid').apply(lambda x: self.shift_label(x, 'Intervention'))
        model_auprc = {}
        predictions_df = pd.DataFrame()
        for model_name, model in self.models.items():
            model.fit(X_train, y_train)
            self.plot_cm(model_name, model, X_test, y_test)
            try:
                self.calc_shap_values(model_name, model, X_train) # calculate shap values for each model
            except Exception as error:
                pass
            y_pred = model.predict(X_test)
            precision,recall,  _ = precision_recall_curve(y_test, y_pred) # calculate recall and precision
            AUPRC = auc(recall, precision) # calculate area under the precision-recall curve
            model_auprc[model_name] = AUPRC
            print(AUPRC)
            df = pd.DataFrame()
            df['y_test'] = y_test
            df['y_pred'] = y_pred
            df['Intervention'] = test_df.Intervention.values
            df['Model'] = model_name
            predictions_df = predictions_df.append(df)
        self.top_features = self.get_top_features(model_auprc)
        self.predictions_df = predictions_df

    def predict(self):
        if isinstance(self.keep_top_features, int):
            self.initial_prediction()
            self.df = self.df[[self.label_name] + list(self.top_features)]
        self.plot_auc()



    def combine_df(self):
        '''
        Combine the observational, the static and pharmacological data. Choose features using feature_names. Number of pharmacological variables can be reduced by specifying a pharma quantile. The number of pharmacological measurements are listed per variable, if the variable has measurement fewer than the quantile then they will be excluded.
        '''
#         printProgress(sys._getframe(  ).f_code.co_name)
        df_list = []
        for group_how in self.group_how_list:
            df_list.append(pd.read_parquet(self.prepared_path + f'df_{group_how}.parquet'))
        df = pd.concat(df_list)
        columns = df.describe().T.drop_duplicates().T.columns
        df = df[columns]
        static_df = pd.read_parquet(self.preprocessed_path + 'static_df.parquet')

        pharma_df = pd.read_parquet(self.prepared_path + f'pharma_df.parquet')
        self.static_variables = list(static_df.columns.values)
        self.pharma_variables = list(pharma_df.columns.values)
        self.obs_variables = list(df.drop(columns = [self.label_name], errors = 'ignore').columns.values)
        df = df.reset_index().merge(static_df, on = 'patientid', how = 'left') # combine observation and static data
        self.time = [c for c in df.reset_index() if c.endswith('_since_adm')][0] # time index
        df = df.merge(pharma_df, on = ['patientid', self.time], how = 'left') # merge the pharma data to the other data
        df = df.set_index(['patientid', self.time]) # make the index of the data the patient id and the minutes or hours since admission
        self.feature_names = list(df.drop(columns = [self.label_name], errors = 'ignore').columns.values)
        df = df[self.feature_names + [self.label_name]] # selecting the features
        df[pharma_df.columns] = df[pharma_df.columns].fillna(0) # binary negative flag for the pharmacological data after it has been joined onto the rest of the data
        df['Intervention'] = df[self.intervention_features].max(axis = 1).ne(0).astype(int)
        df['time_since_admission'] = df.reset_index()[self.time].values
        self.pharma_df = df[self.pharma_variables]
        df = df[self.obs_variables + self.static_variables + [self.label_name,'Intervention']]
        self.feature_names = self.obs_variables + self.static_variables
        df = self.add_missingness_flag(df, self.obs_variables)
        return df

    def add_missingness_flag(self,df,variables):
        for column in variables:
            df[column + '_missing_flag'] = 0
            df.loc[df[column].isna(), column + '_missing_flag'] = 1
        return df


    def get_X_y(self, df):
        '''
        Seperates the data and the labels
        '''
        if len(self.base_features) == 0:
            X_base = df[self.feature_names].values
        else:
            X_base = df[self.base_features].values
        X = df[self.feature_names].values
        y = df[self.label_name].astype(int).values
        return X, X_base, y


    def interpolate_values(self, X):
        '''interpolates missing values'''
        imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value = 0)#, add_indicator = True)
        X = imputer.fit_transform(X)
        return X


    def evaluate(self, model_name, train, test):
        '''
        Evaluates the models using training and testing data
        '''
        model = self.models[model_name] # chooses model
        if isinstance(model, DecisionTreeClassifier): # decision tree classifier is the base model so uses the base variables
            X = self.X_base
        else:
            X = self.X
        y = self.y
        scaler = MinMaxScaler(feature_range=(-1, 1))#StandardScaler() # scale the data
        X[train]  = scaler.fit_transform(X[train]) # scale the training data
        X[test] = scaler.transform(X[test]) # scale the testing data
        X[train] = self.interpolate_values(X[train]) # interpolate or impute values
        X[test] = self.interpolate_values(X[test]) # interpolate or impute values
        if isinstance(model,KerasClassifier): # converts keras model to a scikitlearn style model
            history = model.fit(X[train], y[train], epochs = 50,
                          callbacks=[EarlyStopping('val_auprc', patience=5, mode = 'max')], verbose = 0,  validation_split = 0.1) # fit the keras model
        else:
            model.fit(X[train], y[train]) # fit other types of model
        y_pred = model.predict_proba(X[test]) # predict value using the trained machine learning models
        if not isinstance(self.keep_top_features, int):
            self.plot_cm(model_name, model, X[test], y[test])
            try:
                self.calc_shap_values(model_name, model, X[train]) # calculate shap values for each model
            except Exception as error:
                print(error)

        fpr, tpr, _ = roc_curve(y[test], y_pred[:,1]) # calculate false postive and true positive rate
        interp_tpr = np.interp(self.mean_vals, fpr, tpr) #
        interp_tpr[0] = 0.0

        AUC = auc(tpr, fpr) # calculate area under the roc curve
        precision,recall,  _ = precision_recall_curve(y[test], y_pred[:,1]) # calculate recall and precision
        interp_recall = np.interp(self.mean_vals, precision, recall)
        interp_recall[-1] = 0.0
        AUPRC = auc(recall, precision) # calculate area under the precision-recall curve
        return interp_tpr, interp_recall, AUC, AUPRC


    def cross_validate(self, model_name):
        '''
        cross valudate every model for specified number of folds
        '''
        tprs = [] # tpr for each fold
        aucs = [] # auc for each fold
        auprcs = [] # auprc for each fold
        precisions = [] # precision for each fold

        for i, (train, test) in enumerate(self.cv.split(self.X, self.y)): # loop through each fold
            tpr, precision, AUC, AUPRC = self.evaluate(model_name,train, test) # evaluate fold
            self.train = train
            self.test = test
            # add evaluation metrics of fold to the rest
            tprs.append(tpr)
            aucs.append(AUC)
            precisions.append(precision)
            auprcs.append(AUPRC)
        # calculate values using metrics for plotting
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(self.mean_vals, mean_tpr)
        std_auc = np.std(aucs)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        mean_precision = np.mean(precisions, axis=0)
        mean_precision[0] = 1.0
        mean_auprc = auc(self.mean_vals, mean_precision)
        std_auprc = np.std(auprcs)
        std_precision = np.std(precisions, axis=0)
        precisions_upper = np.minimum(mean_precision + std_precision, 1)
        precisions_lower = np.maximum(mean_precision - std_precision, 0)
        tpr_df = pd.DataFrame([tprs_lower, mean_tpr, tprs_upper], index = ['tprs_lower','mean_tpr','tprs_upper']).T
        precision_df = pd.DataFrame([precisions_lower, mean_precision, precisions_upper], index = ['precisions_lower','mean_precision','precisions_upper']).T
        auc_df = pd.DataFrame([aucs, auprcs], index = ['auc','auprc']).T
        save_data(tpr_df, self.results_path + f'{model_name}/tpr.csv')
        save_data(precision_df, self.results_path + f'{model_name}/precision.csv')
        save_data(auc_df, self.results_path + f'{model_name}/auc.csv')
        return mean_tpr,tprs_lower, tprs_upper, mean_precision, precisions_lower, precisions_upper, mean_auc, std_auc, mean_auprc, std_auprc


    def plot_auc(self):
        '''
        Plotting the area under the ROC curve and precision-recall curve for each fold for each model
        '''
        fig, axs = plt.subplots(1, 2, figsize = (15,6)) # create a figure for both AUC plots
        ax_roc, ax_pr = axs # split the plots
        for model_name, model in self.models.items(): # loop through model
#             printProgress('Evaluating: ' +model_ name)
            tic = datetime.now() # start timer
            mean_tpr,tprs_lower, tprs_upper, mean_precision, precisions_lower, precisions_upper, mean_auc, std_auc, mean_auprc, std_auprc = self.cross_validate(model_name) # evaluation metrics for each model cross val

            ax_roc.plot(self.mean_vals, mean_tpr, color=self.colors[model_name],
                        label=f'{model_name} (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})', lw=3) # plot the AUROC
            ax_roc.fill_between(self.mean_vals, tprs_lower, tprs_upper, color=self.colors[model_name], alpha=.2) # plot the standard deviation of AUROC
            ax_pr.plot(self.mean_vals, mean_precision, color=self.colors[model_name],
                    label=f'{model_name} (AUPRC = {mean_auprc:.2f} $\pm$ {std_auprc:.2f})',
                    lw=2, alpha=.8)    # plot the AUPRC
            ax_pr.fill_between(self.mean_vals, precisions_lower, precisions_upper, color=self.colors[model_name], alpha=.2) # plot the standard deviation of AUPRC
            toc = datetime.now() # stop timer
            time_taken = (toc-tic).total_seconds()/60
            print(f'\nIt took {time_taken:.2f} minutes to evaluate {model_name}')

        # plot aesthetics
        ax_roc.plot([0, 1], [0, 1], linestyle='--', lw=2, color='#595959ff',
            label='No Skill')
        ax_roc.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               title="Receiver operating characteristic curve",
                xlabel = 'False Positive Rate',
                  ylabel = 'True Positive Rate')
        ax_roc.legend(loc="lower right")
        ax_pr.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               title="Precision-recall curve",
                 xlabel = 'Recall',
                 ylabel = 'Precision')
        ax_pr.legend(loc="upper right")
        save_data(fig, self.results_path + 'AUC.png')
#         fig.savefig(results_path + 'AUC.png',bbox_inches='tight') # save the figure






    def calc_shap_values(self,model_name, model, train_data):
        '''
        Calcuates shap values
        '''
        if model_name not in self.models:
            print(f'{model_name} not in list of models')
        else:
            if model_name not in self.model_shap_values:
                explainer = shap.Explainer(model)
                shap_values = explainer(train_data)
                shap_values.feature_names  = self.feature_names
                if len(shap_values.shape) == 3:
                    shap_values.values = shap_values.values[:,:,1]
                    shap_values.base_values = shap_values.base_values[:,1]

                self.model_shap_values[model_name] = shap_values


    def plot_feature_importances (self, model_name):
        '''
        Plots shap values
        '''
        fig, ax = plt.subplots(figsize=(20, 10))
#         self.feature_importances = {}
        shap_values = self.model_shap_values[model_name]
        shap.plots.beeswarm(shap_values, max_display=20, show = False)
        save_data(fig, self.results_path + f'{model_name}/shap.png')

    def plot_patient(self, pid):
        fig, ax = plt.subplots(figsize=(20, 10))
        features = list(set(self.top_features).intersection(set(self.obs_variables)))
        df = self.df.loc[pid][[self.label_name] + features]
        if self.label_name in df.columns:
            case_times = df.loc[df[self.label_name] == 1].index
            for time in case_times:
                 plt.axvline(time, color = '#0097A9')
            intervention_times = df.loc[df['Intervention'] == 1].index
            for time in intervention_times:
                 plt.axvline(time, color = '#0097A9')
        for variable in df.columns:
            if variable != self.label_name or variable != 'time_since_admission':
                plt.plot(df.index,df[variable], label = variable)
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
