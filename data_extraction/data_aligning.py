import pandas as pd
from datetime import time, timedelta, date
import yfinance as yf
from pathlib import Path

stock_dict = pd.read_excel("stock_dict.xlsx")
def extracting_prices(stock_name):
    stock_tick = stock_dict.loc[stock_dict["Company Name"] == stock_name, "Stock Name"].values[0]
    print("stock_tick",stock_tick)
    data = yf.download(stock_tick, period='3y', multi_level_index=False)
    data.to_csv(f"Data/raw/yahoo/{stock_name}_yahoo.csv")

price_base_path = "Data/raw/yahoo"
news_base_path = "Data/processed/extracted_news"


def aligning_csv(stock_name):
    price_path = price_base_path+"/"+stock_name+"_yahoo.csv"
    news_path = news_base_path+"/"+stock_name+".csv"
    price_df = pd.read_csv(price_path)
    news_df = pd.read_csv(news_path)

    # ======================
    # DATETIME PROCESSING
    # ======================
    news_df["news_datetime"] = pd.to_datetime(news_df["news_datetime"])
    latest_date = news_df["news_datetime"][0]
    price_df["Date"] = pd.to_datetime(price_df["Date"])

    # normalize price date
    price_df["trade_date"] = price_df["Date"].dt.date

    # set index for fast lookup
    price_df.set_index("trade_date", inplace=True)

    # set of valid trading days
    trading_days = set(price_df.index)


    # ======================
    # MARKET CLOSE TIME
    # ======================
    market_close = time(15, 30) # time - 15:30


    # ======================
    # FUNCTION: MAP NEWS → TRADING DAY
    # ======================

    # will find the price date with the corresponding news date
    def get_event_date(news_dt):

        news_date = news_dt.date()

        # if news released after market close → shift to next day
        # print(news_dt.date())
        if news_dt.time() > market_close:
            # print(news_dt.time())
            # print(market_close)
            news_date += timedelta(days=1)

        # move forward until valid trading day
        while news_date not in trading_days:
            news_date += timedelta(days=1)

        return news_date


    # ======================
    # FUNCTION: GET FUTURE PRICE
    # ======================

    # will give the prices
    def get_future_price(date, offset):

        d = date
        count = 0

        while count < offset:
            d += timedelta(days=1)
            if d in trading_days:
                count += 1

        return price_df.loc[d]["Close"]


    # ======================
    # ALIGNMENT LOOP
    # ======================

    records = []
    missed_urls = []
    count = 0
    for _, row in news_df.iterrows():

        news_dt = row["news_datetime"]
        # print(count)
        try:
            event_date = get_event_date(news_dt) # getting the price date of yahoo finance
        except ValueError:
            print("Skipping entry: returned None")
            missed_urls.append({"url" : row["link"],
                                "error" : "returned None"})
            continue
        if event_date not in price_df.index:
            print("Skipping entry: News date not in pride df")
            missed_urls.append({"url" : row["link"],
                               "error" : "news date not matched with price date"})
            continue

        # print(event_date)
        try:
            if latest_date.date() - event_date < timedelta(days=3):
                continue
        except AttributeError:
            print("Skipping entry: news_datetime is None")
            missed_urls.append(row["link"])
            continue 


        close_T = price_df.loc[event_date]["Close"]

        # future prices
        close_T1 = get_future_price(event_date, 1)
        close_T2 = get_future_price(event_date, 2)
        close_T3 = get_future_price(event_date, 3)

        # returns
        r1 = (close_T1 - close_T) / close_T
        r2 = (close_T2 - close_T) / close_T
        r3 = (close_T3 - close_T) / close_T

        records.append({
            "news_id": row["news_id"],
            "headline": row["headline"],
            "news_time": news_dt,
            "event_date": event_date,
            "close_T": close_T,
            "ret_1d": r1,
            "ret_2d": r2,
            "ret_3d": r3
        })
        count += 1

    # ======================
    # FINAL DATAFRAME
    # ======================

    aligned_df = pd.DataFrame(records)

    # print(aligned_df.head)

    # save
    aligned_df.to_csv(F"Data/processed/aligned/{stock_name}_aligned.csv", index=False)