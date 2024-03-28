import pandas as pd
import json

def load_json(path: str) -> pd.DataFrame:
    with open(path, 'r') as f:
        json_info = json.load(f)
        if 'data' in json_info.keys():
            return pd.DataFrame.from_dict(json_info['data']) 
        else:
            print("Wrong data structure, json file doesn't have data key")
            return pd.DataFrame([])



def create_phase_dataset(all_info_df: pd.DataFrame, phase_name: str, days_before: int, duration: int, func: str = 'max') -> pd.DataFrame:
    """
    This function make shift by days before to merge phase with the last one. 
    It makes window function of each column with window equalse duration of this phase
    :param all_info_df: pd.DataFrame with date column and all information about phase
    :param phase_name: string name of harvest phase
    :param days_before: days until the harvest has been harvested
    :param duration: phases duration
    """
    
    columns = [f'counts_{phase_name}', f'avg_areas_{phase_name}']
    columns_to_extend = [[col for col in all_info_df.columns if func in col]]
    columns.extend(sensor_statistic.columns)
    new_df = all_info_df[columns].copy()
    new_df = new_df.shift(days_before)
    # add date for merge
    new_df['date'] = pd.to_datetime(all_info_df['date'])
    new_df = new_df.sort_values('date')
    new_df = new_df.set_index('date')
    new_df = new_df.dropna()
    
    # make mean, max and sum window function
    result_df = new_df.rolling(window=duration).agg(func)
    result_df.columns = [f'{phase_name}_{func}_' + col for col in result_df.columns] 
    return result_df
