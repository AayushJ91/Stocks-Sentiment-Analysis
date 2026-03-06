from data_extraction.headlines import headlines_extractor_playwright_asyncio
import asyncio

airtle_url = 'https://www.moneycontrol.com/news/tags/airtel/news/'

list_airtel = asyncio.run(headlines_extractor_playwright_asyncio(airtle_url))

print(list_airtel)