# import n
import pandas as pd
import numpy as np
import requests as req
from bs4 import BeautifulSoup as soup
import pickle
from concurrent.futures import ThreadPoolExecutor

count = 0

def main():
    data = pd.read_csv("~/../../mnt/c/Users/Ethan/Downloads/archive/hacker_news_sample.csv")
    urls = data.url
    types = data.type

    with ThreadPoolExecutor(max_workers=100) as p:
        htlm_texts = p.map(url_pipeline, zip(urls, types))

    with open("html content", "wb") as fb:
        pickle.dump(htlm_texts, fb)


def url_pipeline(url_type):
    url = url_type[0]
    type = url_type[1]

    try:
        if type == "comment":
            return ""

        if isinstance(url, str):
            html = fetch_html(url)
            data = extract_data(html)
            return data
        return ""
    except:
        return ""
            


def fetch_html(url):
    request = req.get(url)
    return request.text


def extract_data(html):
    if html == "":
        return html
    html_soup = soup(html, 'html.parser')
    return html_soup.get_text()
    

if __name__ == "__main__":
    main()