from datetime import date, datetime
import os
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
import time
# Make sure to install playwright browsers by running playwright install on the command line or !playwright install from Jupyter
import pandas as pd
import asyncio


async def get_html(url, selector, sleep=5, retries=3):
    html = None
    for i in range(1, retries+1):
        time.sleep(sleep * i)
        try:
            async with async_playwright() as p:
                browser = await p.firefox.launch()
                page = await browser.new_page()
                await page.goto(url)
                print(await page.title())
                html = await page.inner_html(selector)
        except PlaywrightTimeout:
            print(f"Timeout error on {url}")
            continue
        else:
            break
    return html


# directory = STANDINGS_DIR
async def scrape_season(season, directory):
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    html = await get_html(url, "#content .filter")

    soup = BeautifulSoup(html)
    links = soup.find_all("a")
    standings_pages = [f"https://www.basketball-reference.com{l['href']}" for l in links]

    for url in standings_pages:
        save_path = os.path.join(directory, url.split("/")[-1])
        if os.path.exists(save_path):
            continue

        html = await get_html(url, "#all_schedule")
        with open(save_path, "w+") as f:
            f.write(html)


# directory = SCORES_DIR
async def scrape_game(standings_file, directory):
    # standings file: single month in a season table
    with open(standings_file, 'r') as f:
        html = f.read()

    soup = BeautifulSoup(html)
    links = soup.find_all("a")
    hrefs = [l.get('href') for l in links]
    box_scores = [f"https://www.basketball-reference.com{l}" for l in hrefs if l and "boxscore" in l and '.html' in l]

    # box scores contains all of the box scores from a particular month in a season
    for url in box_scores:
        save_path = os.path.join(directory, url.split("/")[-1])
        if os.path.exists(save_path):
            continue

        html = await get_html(url, "#content")
        if not html:
            continue
        with open(save_path, "w+") as f:
            f.write(html)

async def main():
    current_year = date.today().year

    SEASONS = list(range(current_year - 7, current_year + 1))

    DATA_DIR = "data"
    PREDICTIONS_DIR = os.path.join(DATA_DIR, "to_predict")
    STANDINGS_DIR = os.path.join(DATA_DIR, "standings")
    SCORES_DIR = os.path.join(DATA_DIR, "scores")

    for season in SEASONS:
        await scrape_season(season, STANDINGS_DIR)

    standings_files = os.listdir(STANDINGS_DIR)

    standings_files = [s for s in standings_files if '.html' in s]

    for f in standings_files:
        filepath = os.path.join(STANDINGS_DIR, f)

        await scrape_game(filepath, SCORES_DIR)

if __name__ == "__main__":
    asyncio.run(main())
