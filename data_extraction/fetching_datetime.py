import requests
from bs4 import BeautifulSoup
from requests.exceptions import TooManyRedirects
import time
from datetime import datetime
import pytz
import uuid
import json
import csv
from pathlib import Path

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
    "Referer": "https://www.moneycontrol.com/"
}

session = requests.Session()
session.headers.update(HEADERS)

IST = pytz.timezone("Asia/Kolkata")

def extract_moneycontrol_date(soup):
    schedule_div = soup.find("div", class_="article_schedule")

    if not schedule_div:
        return None, None

    # 1️⃣ Extract date part
    date_span = schedule_div.find("span")
    if not date_span:
        return None, None

    date_part = date_span.text.strip()   # "February 04, 2026"

    # 2️⃣ Extract time part (text after '/')
    full_text = schedule_div.get_text(separator=" ").strip()
    # Example: "February 04, 2026 / 18:17 IST"

    try:
        time_part = full_text.split("/")[-1].strip()  # "18:17 IST"
    except IndexError:
        return None, None

    # 3️⃣ Combine
    raw_date_text = f"{date_part} {time_part}"
    # "February 04, 2026 18:17 IST"

    # 4️⃣ Parse
    try:
        dt = datetime.strptime(
            raw_date_text.replace(" IST", ""),
            "%B %d, %Y %H:%M"
        )
        news_datetime = IST.localize(dt)
    except ValueError:
        return raw_date_text, None

    return raw_date_text, news_datetime


def fetch_article_soup(url):
    try:
        response = session.get(
            url,
            timeout=10,
            allow_redirects=True
        )
        # print(response.status_code)
        if response.status_code != 200:
            return None

        return BeautifulSoup(response.text, "html.parser")

    except TooManyRedirects:
        print("Redirect loop detected:", url)
        return None

    except requests.RequestException as e:
        print("Request failed:", url, e)
        return None
    

def structuring_data(news_list, name):
    # missed_urls = []
    news_records = []
    article_count = 0
    for news in news_list:
        article_count += 1

        article_url = news["link"]
        headline = news["headline"]
        soup = fetch_article_soup(article_url)

        if soup is None:
            news_records.append({
                "news_id": str(uuid.uuid4()),
                "headline": headline,
                "link": article_url,
                "source": "moneycontrol",
                "raw_date_text": None,
                "news_datetime": None,
                "article_text":None,
                "scraped_at": datetime.now(IST).isoformat(),
                "scrape_page": None,
                "status": "fetch_failed"
            })
            continue
        # print("mid")
        # 🔹 extract date from article page
        raw_date_text, news_datetime = extract_moneycontrol_date(soup)

        # print(news_datetime.date())


        record = {
            "news_id": str(uuid.uuid4()),
            "headline": headline,
            "link": article_url,
            "source": "moneycontrol",
            "raw_date_text": raw_date_text,
            "news_datetime": (
                news_datetime.isoformat()
                if news_datetime else None
            ),
            "article_text":None,
            "scraped_at": datetime.now(IST).isoformat(),
            "scrape_page": None,
            "status": "success" if news_datetime else "date_missing"
        }

        news_records.append(record)

        if article_count % 10 == 0:
            print(f"Processed {article_count} articles")

        # print("bottom")

        time.sleep(2)  # polite scraping


    Path('Data/raw/moneycontrol/').mkdir(parents=True, exist_ok=True)
    with open(f'Data/raw/moneycontrol/{name}.json','w', encoding='utf-8') as f:
        json.dump(news_records, f, indent=10, ensure_ascii=False)

    with open(f'Data/raw/moneycontrol/{name}.json') as file:
        d = json.load(file)



def jsonTocsv(name):
    with open(f'Data/raw/moneycontrol/{name}.json') as file:
        d = json.load(file)
    Path('Data/processed/extracted_news/').mkdir(parents=True, exist_ok=True)
    present_df = open(f'Data/processed/extracted_news/{name}.csv', "w", newline='')
    cw = csv.writer(present_df)
    c = 0
    for data in d:
        if c == 0:
            header = data.keys()
            cw.writerow(header)
            c += 1
        cw.writerow(data.values())

    present_df.close()