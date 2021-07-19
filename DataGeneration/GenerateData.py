import yfinance as yf
from raw_to_gafs_funcs import clean_non_trading_times, set_gaf_data, convert_to_gaf_and_save

def generate(stock: str, start: str, end: str):
    """
    apply sequentially all the function necessary to create train and test sets

    :param stock: Ticker of your stock
    :param start: start date
    :param end: end date
    :return:
    """
    data_ticker = yf.Ticker(stock)
    data = data_ticker.history(start=start, end=end, interval="1h")
    data = clean_non_trading_times(data)
    decision_map,n = set_gaf_data(data)
    convert_to_gaf_and_save(decision_map, n, test_split=0.3)

if __name__=="__main__":
    stock = "GOOGL"
    start="2020-07-01"
    end = "2021-06-01"

    generate(stock,start,end)