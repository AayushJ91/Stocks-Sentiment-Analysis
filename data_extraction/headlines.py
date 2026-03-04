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


mc_base_url = "https://www.moneycontrol.com/news/tags/reliance-industries/news/"
stock_name = "RI"
year = "2026"

url = f"https://www.moneycontrol.com/stocks/company_info/stock_news.php?sc_id={stock_name}&durationType=Y&Year={2026}"

def headlines_extractor(url):
    page = 1
    all_news = []
    seen_links = set()

    while True:
        if page == 1:
            using_url = url + "/"
        else:
            using_url = f"{url}/page-{page}/"

        print(f"Scraping page {page}: {using_url}")

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