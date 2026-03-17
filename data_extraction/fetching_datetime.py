import requests
from bs4 import BeautifulSoup
from requests.exceptions import TooManyRedirects
import time
import os
from datetime import datetime
import pytz
import uuid
import json
import csv
import logging
from pathlib import Path
from .logger import setup_logger

logger = setup_logger("data_fetcher")

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
    """
    Extracts the publication date and time from a Moneycontrol article page.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content of the article page
    
    Returns:
        tuple: (raw_date_text, news_datetime) where:
            - raw_date_text (str or None): The raw date string as found in the page
            - news_datetime (datetime or None): Parsed datetime object in IST timezone, or None if parsing fails
    """
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
    """
    Fetches an article from the given URL and returns its parsed HTML content.
    
    Args:
        url (str): The URL of the article to fetch
    
    Returns:
        BeautifulSoup or None: Parsed HTML as BeautifulSoup object if successful, None if request fails
    """
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
        logger.warning(f"Redirect loop detected: {url}")
        return None

    except requests.RequestException as e:
        logger.error(f"Request failed: {url} - {e}")
        return None
    

def structuring_data(news_list, name):
    """
    Processes a list of news articles: fetches each article and extracts its publication date/time.
    
    Args:
        news_list (list): List of news dictionaries containing 'link' and 'headline' keys
        name (str): Identifier for the news source (used for logging)
    
    Returns:
        list: List of structured news record dictionaries containing news_id, headline, link, 
              source, timestamps, and status for each article.
    
    Processing Details:
        - Fetches each article URL and extracts publication date/time
        - Handles failures (fetch errors, missing dates) gracefully with status tracking
        - Records include news_id, headline, link, source, timestamps, and status
        - Includes 2-second delay between requests for polite scraping
    """
    news_records = []
    article_count = 0
    failed_fetches = 0
    date_missing = 0
    
    logger.info(f"Starting to structure {len(news_list)} articles for '{name}'")
    
    for news in news_list:
        article_count += 1

        article_url = news["link"]
        headline = news["headline"]
        soup = fetch_article_soup(article_url)

        if soup is None:
            failed_fetches += 1
            logger.warning(f"Failed to fetch article {article_count}: {article_url}")
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
        # 🔹 extract date from article page
        raw_date_text, news_datetime = extract_moneycontrol_date(soup)

        if not news_datetime:
            date_missing += 1
            logger.debug(f"Date missing for article {article_count}: {headline[:50]}")

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
            logger.info(f"Processed {article_count} articles")

        time.sleep(2)  # polite scraping

    logger.info(f"Completed: {article_count} total, {failed_fetches} failed, {date_missing} missing dates")
    return news_records



def save_news_records_to_csv(news_records, name, csv_path='Data/raw/news/'):
    """
    Converts a list of news records directly to a CSV file.
    
    Args:
        news_records (list): List of news record dictionaries to save
        name (str): Identifier for the news source (used for filename)
        csv_path (str): Directory path where the output CSV file should be saved. 
                       Defaults to 'Data/raw/CSVs/'
    
    Returns:
        None. Saves records to {csv_path}/{name}.csv
    
    Processing Details:
        - Extracts column headers from the first record
        - Writes all records with headers to CSV
        - Handles errors gracefully with logging
    """
    if not news_records:
        logger.warning(f"No records to save for '{name}'")
        return
    
    os.makedirs(csv_path, exist_ok=True)
    logger.info(f"Saving {len(news_records)} records to CSV for '{name}'")
    try:
        csv_file_path = os.path.join(csv_path, f'{name}.csv')
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            cw = csv.writer(csvfile)
            # Write header from first record
            header = news_records[0].keys()
            cw.writerow(header)
            # Write all records
            for record in news_records:
                cw.writerow(record.values())
        
        logger.info(f"Successfully saved CSV to {csv_file_path}")
    except Exception as e:
        logger.error(f"Error saving records to CSV: {e}", exc_info=True)


def jsonTocsv(name, json_path='Data/raw/moneycontrol/', csv_path='Data/raw/CSVs/'):
    """
    Converts a JSON file containing news records to a CSV file.
    
    Args:
        name (str): Identifier for the news source (used for filename)
        json_path (str): Directory path where the input JSON file is located. 
                        Defaults to 'Data/raw/moneycontrol/'
        csv_path (str): Directory path where the output CSV file should be saved. 
                       Defaults to 'Data/processed/extracted_news/'
    
    Returns:
        None. Saves converted records to {csv_path}/{name}.csv
    
    Processing Details:
        - Reads JSON file from {json_path}/{name}.json
        - Extracts column headers from the first record
        - Writes all records with headers to CSV
        - Handles errors gracefully with logging
    """
    os.makedirs(csv_path, exist_ok=True)
    logger.info(f"Converting JSON to CSV for '{name}'")
    try:
        with open(f'{json_path}{name}.json') as file:
            d = json.load(file)
        Path(csv_path).mkdir(parents=True, exist_ok=True)
        present_df = open(f'{csv_path}{name}.csv', "w", newline='')
        cw = csv.writer(present_df)
        c = 0
        for data in d:
            if c == 0:
                header = data.keys()
                cw.writerow(header)
                c += 1
            cw.writerow(data.values())

        present_df.close()
        logger.info(f"Successfully saved CSV to {csv_path}{name}.csv")
    except Exception as e:
        logger.error(f"Error converting JSON to CSV: {e}", exc_info=True)