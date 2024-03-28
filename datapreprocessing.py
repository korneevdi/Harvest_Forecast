import pandas as pd
from xgboost import XGBRegressor

PHASE_FLOWER = 7
PHASE_FRUITSET = 7
PHASE_SMALLGREEN = 8
PHASE_WHITE = 7
PHASE_TURNING = 2
PHASE_RED = 3

FULL_CYCLE = PHASE_FLOWER + PHASE_FRUITSET + PHASE_SMALLGREEN + PHASE_WHITE + PHASE_TURNING + PHASE_RED
FRUITSET_DAY = FULL_CYCLE - PHASE_FLOWER
SMALLGREEN_DAY = FRUITSET_DAY - PHASE_FRUITSET
WHITE_DAY = SMALLGREEN_DAY - SMALLGREEN_DAY
TURNING_DAY = PHASE_TURNING + PHASE_RED

        

class DataPreprocessing:
    
    def __init__(self, sensor_df: pd.DataFrame, image_df: pd.DataFrame, entries_df: pd.DataFrame):
        self._sensor_df = self.sensor_preprocess(sensor_df)
        self._image_df = self.image_preprocess(image_df)
        self._entries_df = self.etries_preprocess(entries_df)

    @property
    def sensor_df(self):
        return self._sensor_df
        
    @property
    def image_df(self):
        return self._image_df
    
    @property
    def entries_df(self):
        return self._entries_df
    
    @staticmethod
    def sensor_preprocess(sensor_df):
        sensor_df['time'] = pd.to_datetime(sensor_df['time'])
        sensor_df = sensor_df.sort_values('time')
        sensor_df['hour'] = sensor_df['time'].dt.hour
        sensor_df['date'] = sensor_df['time'].dt.date
        sensor_df = sensor_df.drop(columns=['time'])
        sensor_df = sensor_df.dropna(subset='Air_Humidity')
        
        return sensor_df

    @staticmethod
    def image_preprocess(image_df):
        image_df['time'] = pd.to_datetime(image_df['time'])
        image_df['date'] = image_df['time'].dt.date
        image_df = image_df.drop(columns='time')
        return image_df

    @staticmethod
    def etries_preprocess(entries_df):
        entries_df = pd.json_normalize(entries_df['entries'])
        entries_df['date'] = entries_df['taken_at'].apply(lambda x: pd.to_datetime(x,unit='s')).dt.date
        entries_df = entries_df.drop(columns=['taken_at'])
        return entries_df
        
    def fill_na_sensosors_with_xgb(self):
        """
        This function fill nan values by using xgb connections between Air_Humidity, Air_Temperature and Time
        """
        # use this approach to find nan values
        data_for_par_missing_model = self._sensor_df.dropna(subset=['Air_Humidity','Air_Temperature', 'PAR'])
        feature = data_for_par_missing_model[['Air_Humidity', 'Air_Temperature',]]
        target = data_for_par_missing_model[['PAR']]
        
        nan_par = self._sensor_df[self._sensor_df['PAR'].isna()].reset_index(drop=True)
        nan_par_features = nan_par[['Air_Humidity', 'Air_Temperature',]]
    
        
        # make regression by all data
        reg = XGBRegressor(random_state=42)
        model = reg.fit(feature, target)
        predict = model.predict(nan_par_features)
        
        # fillna
        self._sensor_df.loc[self._sensor_df['PAR'].isna(), 'PAR'] = predict
    
    def find_sensor_statistic(self, agg_func = ['sum', 'max', 'min', 'mean']):
        sensor_statistic = self._sensor_df.drop(columns=['hour']).groupby('date').agg(agg_func)
        sensor_statistic.columns = [col[0] + '_' + col[1] for col in sensor_statistic.columns]
        sensor_statistic = sensor_statistic.reset_index()
        return sensor_statistic

    def create_phase_dataset(self, all_info_df: pd.DataFrame,
                             sensor_statistic: pd.DataFrame,
                             phase_name: str,
                             days_before: int,
                             duration: int) -> pd.DataFrame:
        """
        This function makes shift by days before merging phase with the last one. 
        It makes window function of each column with window equalse duration of this phase
        :param all_info_df: pd.DataFrame with date column and all information about phase
        :param phase_name: string name of harvest phase
        :param days_before: days until the harvest has been harvested
        :param duration: phases duration
        """
        new_df = all_info_df.copy()
        new_df = new_df.sort_values('date')
        new_df = new_df.set_index('date')
        new_df = new_df.ffill()
        mean_df = self.make_rolling(new_df, sensor_statistic, phase_name, 'mean', window=duration, )
        return mean_df.shift(days_before)

    @staticmethod
    def make_rolling(df_to_roll, sensor_statistic, phase_name, agg_func, window):
        df = df_to_roll.copy()
        columns = [ f'counts_{phase_name}', f'avg_areas_{phase_name}']
        columns.extend(sensor_statistic.columns[1:])
        # make mean, max and sum window function
        df = df[columns]
        df = df.rolling(window=window).agg(agg_func)
        df.columns = [f'{phase_name}_' + col if phase_name not in col else col for col in df.columns ] 
        return df

    @staticmethod
    def make_df_for_model(df1: pd.DataFrame, df2: pd.DataFrame, cur_phase: str):
        df = df1.merge(df2, left_index=True, right_index=True).copy()
        df = df.dropna()
        df['target'] = df[f'counts_{cur_phase}'] * df[f'avg_areas_{cur_phase}']
        df = df.drop(columns=[f'counts_{cur_phase}', f'avg_areas_{cur_phase}'])

        df['day_since_start'] = [i for i in range(df.shape[0])]
        return df.reset_index()

    def processing(self):
        self.fill_na_sensosors_with_xgb()
        sensor_statistic = self.find_sensor_statistic()
        all_info_df = self.image_df.merge(sensor_statistic, on='date', how='inner')
        all_info_df = self.entries_df[['date', 'value']].merge(all_info_df, on='date', how='outer')
        
        flower_df = self.create_phase_dataset(all_info_df,sensor_statistic, 'flower', FULL_CYCLE, PHASE_FLOWER )
        fruitset_df = self.create_phase_dataset(all_info_df, sensor_statistic,  'fruitset', FRUITSET_DAY, PHASE_FRUITSET )
        smallgreen_df = self.create_phase_dataset(all_info_df, sensor_statistic, 'smallgreen', SMALLGREEN_DAY, PHASE_SMALLGREEN )
        white_df = self.create_phase_dataset(all_info_df, sensor_statistic,  'white', WHITE_DAY, PHASE_WHITE )
        turning_df = self.create_phase_dataset(all_info_df, sensor_statistic, 'turning', TURNING_DAY, PHASE_TURNING)
        red_df = self.create_phase_dataset(all_info_df, sensor_statistic, 'red', PHASE_RED, PHASE_RED)

        
        #for flower we don't have prev phase
        flower_final_df = self.make_df_for_model(flower_df, flower_df.reset_index()[['date']].set_index('date'), 'flower')
        fruitset_final_df = self.make_df_for_model(fruitset_df, flower_df, 'fruitset')
        smallgreen_final_df = self.make_df_for_model(smallgreen_df, fruitset_df, 'smallgreen')
        white_final_df = self.make_df_for_model(white_df, smallgreen_df, 'white')
        turning_final_df = self.make_df_for_model(turning_df, white_df,'turning')
        red_final_df = self.make_df_for_model(red_df, turning_df, 'red')

        return flower_final_df, fruitset_final_df, smallgreen_final_df, white_final_df, turning_final_df, red_final_df
