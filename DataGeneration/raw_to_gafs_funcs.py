import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pandas.tseries.holiday import USFederalHolidayCalendar as Calendar
from data.GramianAngularField import fit_transform


def clean_non_trading_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    :param df: Data with weekends and holidays
    :return trading_data:
    """
    df.reset_index(inplace=True)
    df.rename(columns={"index": 'Date'}, inplace=True)

    # Weekends go out
    df = df[df['Date'].dt.weekday < 5].reset_index(drop=True)
    df = df.set_index('Date')
    # Remove non trading hours
    df = df.between_time('9:00', '16:00')
    df.reset_index(inplace=True)
    # Holiday days we want to delete from DataGeneration
    holidays = Calendar().holidays(start='2000-01-01', end='2020-12-31')
    m = df['Date'].isin(holidays)
    clean_df = df[~m].copy()
    trading_data = clean_df.fillna(method='ffill')
    return trading_data


def trading_action(future_close: float, current_close: float) -> str:
    if future_close < current_close:
        decision = 'LONG'
    else:
        decision = 'SHORT'
    return decision


def set_gaf_data(df):
    """
    :param df: DataFrame DataGeneration
    :return: None
    """
    dates = df['Date'].dt.date
    dates = dates.drop_duplicates()
    list_dates = dates.apply(str).tolist()
    index = 20  # rows of DataGeneration used on each GAF
    # Container to store DataGeneration for the creation of GAF
    decision_map = {key: [] for key in ['LONG', 'SHORT']}
    while True:
        if index >= len(list_dates) - 1:
            break
        # Select appropriate timeframe
        data_slice = df.loc[(df['Date'] > list_dates[index - 20]) & (df['Date'] < list_dates[index])]
        gafs = []
        # Group data_slice by time frequency
        for freq in ['1h', '2h', '4h', '1d']:
            group_dt = data_slice.groupby(pd.Grouper(key='Date', freq=freq)).mean().reset_index()
            group_dt = group_dt.dropna()
            gafs.append(group_dt['Close'].tail(20))
        # Decide what trading position we should take on that day
        future_value = df[df['Date'].dt.date.astype(str) == list_dates[index]]['Close'].iloc[-1]
        current_value = data_slice['Close'].iloc[-1]
        decision = trading_action(future_close=future_value, current_close=current_value)
        decision_map[decision].append([list_dates[index - 1], gafs])
        index += 1
    return decision_map


def convert_to_gaf_and_save(decision_map: dict):
    os.makedirs("SHORT", exist_ok=True)
    os.makedirs("LONG", exist_ok=True)

    slots = [[(0, 20), (0, 20)], [(0, 20), (20, 40)], [(20, 40), (0, 20)], [(20, 40), (20, 40)]]
    for decision in decision_map:
        for day in decision_map[decision]:
            index = day[0]
            img = np.zeros((40, 40))
            for i in range(len(day[1])):
                img[slots[i][0][0]:slots[i][0][1], slots[i][1][0]:slots[i][1][1]] = fit_transform(day[1][i])
            plt.pcolormesh(img, cmap='RdBu')
            plt.savefig(decision + "/{}.png".format(index))