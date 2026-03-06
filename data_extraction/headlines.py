import requests
from bs4 import BeautifulSoup, Comment
# Extracting the links of every page for a stock from moneycontrol

HEADERS_PAGES = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
    "Referer": "https://www.moneycontrol.com/"
}

session = requests.Session()
session.headers.update(HEADERS_PAGES)


def headlines_extractor(url):
    page = 1
    all_news = []
    seen_links = set()

    while True:
        if page == 1:
            using_url = url+"/"
        else:
            using_url = f"{url}/page-{page}/"

        print(f"Scraping page {page}: {using_url}")
        # print(f"hi")

        response = session.get(using_url, timeout=10)

        if response.status_code != 200:
            print("Stopping — bad response")
            break

        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.find_all("li", class_="clearfix")

        if not articles:
            print("No articles found. Stopping.")
            break

        new_found = False

        for article in articles:
            headline_tag = article.find("h2")
            link_tag = article.find("a", href=True)

            if headline_tag and link_tag:
                link = link_tag["href"]

                if link not in seen_links:
                    seen_links.add(link)
                    new_found = True

                    all_news.append({
                        "headline": headline_tag.text.strip(),
                        "link": link
                    })

        if not new_found:
            print("No new articles. Reached last page.")
            break

        page += 1

    return all_news


from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup


def headlines_extractor_playwright(base_url):

    all_news = []
    seen_links = set()
    page = 1

    with sync_playwright() as p:

        browser = p.chromium.launch(headless=True)
        page_browser = browser.new_page()

        while True:

            if page == 1:
                url = base_url
            else:
                url = f"{base_url}/page-{page}/"

            print(f"Scraping page {page}: {url}")

            page_browser.goto(url, timeout=60000)

            html = page_browser.content()
            soup = BeautifulSoup(html, "html.parser")

            articles = soup.find_all("li", class_="clearfix")

            if not articles:
                print("No articles found. Stopping.")
                break

            new_found = False

            for article in articles:

                headline_tag = article.find("h2")
                link_tag = article.find("a", href=True)

                if headline_tag and link_tag:

                    link = link_tag["href"]

                    if link not in seen_links:

                        seen_links.add(link)
                        new_found = True

                        all_news.append({
                            "headline": headline_tag.text.strip(),
                            "link": link
                        })

            if not new_found:
                print("Reached last page.")
                break

            page += 1

        browser.close()

    return all_news

import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup


async def headlines_extractor_playwright_asyncio(base_url):

    all_news = []
    seen_links = set()
    page_no = 1

    async with async_playwright() as p:

        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        while True:

            if page_no == 1:
                url = base_url
            else:
                url = f"{base_url}/page-{page_no}/"

            print(f"Scraping page {page_no}: {url}")

            await page.goto(url)

            html = await page.content()

            soup = BeautifulSoup(html, "html.parser")

            articles = soup.find_all("li", class_="clearfix")

            if not articles:
                print("No articles found. Stopping.")
                break

            new_found = False

            for article in articles:

                headline_tag = article.find("h2")
                link_tag = article.find("a", href=True)

                if headline_tag and link_tag:

                    link = link_tag["href"]

                    if link not in seen_links:

                        seen_links.add(link)
                        new_found = True

                        all_news.append({
                            "headline": headline_tag.text.strip(),
                            "link": link
                        })

            if not new_found:
                print("Reached last page.")
                break

            page_no += 1

        await browser.close()

    return all_news