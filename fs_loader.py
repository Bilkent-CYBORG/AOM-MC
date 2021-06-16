import pickle

import dateutil
import pandas as pd
from tqdm import tqdm
# import gmaps
# import gmaps.datasets
import matplotlib.pyplot as plt

"""
This script will parse the Foursquare location checkin dataset and extract the data and nomralize the latitude/longitude
dataset from https://sites.google.com/site/yangdingqi/home/foursquare-dataset
"""

city_name = 'tky'
city_df_name_dict = {'tky': 'fs_tky_df',
                     'nyc': 'fs_nyc_df'}
city_dataset_name_dict = {'tky': 'dataset_tsmc2014/dataset_TSMC2014_TKY.txt',
                          'nyc': 'dataset_tsmc2014/dataset_TSMC2014_NYC.txt'}

saved_df_filename = city_df_name_dict[city_name]  # the name of the file where the dataframe will be saved
file_name = city_dataset_name_dict[city_name]  # the name of the Foursquare  dataset file
load_from_pickle = False  # this is used for debugging, leave it as False

def parse_line(line: str):
    line_arr = line.split("\t")
    user_id = line_arr[0]
    venue_id = line_arr[1]
    venue_cat_id = line_arr[2]
    venue_cat_name = line_arr[3]
    lat = float(line_arr[4])
    long = float(line_arr[5])
    offset = float(line_arr[6])
    time = dateutil.parser.parse(line_arr[7])

    return user_id, venue_id, venue_cat_id, venue_cat_name, lat, long, offset, time


def save_locations(df):
    # save (lat, long) pairs as list
    location_list = df[['lat', 'long']].values.tolist()

    with open(saved_df_filename + '_loc', 'wb') as output:
        pickle.dump(location_list, output, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    colnames = ['user_id', 'venue_id', 'venue_cat_id', 'venue_cat_name', 'lat', 'long', 'offset', 'time']

    if load_from_pickle:
        with open(saved_df_filename, 'rb') as input_file:
            df = pickle.load(input_file)
    else:
        with open(file_name, encoding="ISO-8859-1") as f:
            df = pd.DataFrame([parse_line(l) for l in tqdm(f)], columns=colnames)

        # Normalize lat
        min_lat = df['lat'].min()
        max_lat = df['lat'].max()
        print('lat: min {} max {}'.format(min_lat, max_lat))
        df['lat'] = (df['lat'] - min_lat) / (max_lat - min_lat)

        # Normalize long
        min_long = df['long'].min()
        max_long = df['long'].max()
        print('long: min {} max {}'.format(min_long, max_long))
        df['long'] = (df['long'] - min_long) / (max_long - min_long)

        with open(saved_df_filename, 'wb') as output:
            pickle.dump(df, output, pickle.HIGHEST_PROTOCOL)

    # save_locations(df)
    print('Done loading Foursquare dataset')
