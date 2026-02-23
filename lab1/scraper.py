import os
import csv
import string
from collections import Counter
from datetime import datetime

import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

for _res in ('punkt', 'punkt_tab', 'stopwords'):
    nltk.download(_res, quiet=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def scrape_news(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        elements = soup.find_all(["h2", "h3", "p"], limit=50)
        return [el.get_text(strip=True) for el in elements if len(el.get_text(strip=True)) > 15]
    except Exception as e:
        print(f"[Помилка скрапінгу] {url}: {e}")
        return []


def process_nlp_top5(texts):
    combined_text = " ".join(texts).lower()
    table = str.maketrans("", "", string.punctuation + "«»—–''\"\"")
    cleaned_text = combined_text.translate(table)
    tokens = word_tokenize(cleaned_text)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [w for w in tokens if w not in stop_words and len(w) > 2 and w.isalpha()]
    return Counter(filtered_tokens).most_common(5)


def get_current_time_slot(hour):
    if 5 <= hour < 12:
        return "Ранок: 08:00"
    elif 12 <= hour < 17:
        return "Обід: 13:00"
    else:
        return "Вечір: 19:00"


def main():

    bbc_texts = scrape_news("https://www.bbc.com/news")
    nyt_texts = scrape_news("https://www.nytimes.com")
    all_texts = bbc_texts + nyt_texts

    if not all_texts:
        print("Немає даних для аналізу. Завершення.")
        return

    top_5_terms = process_nlp_top5(all_texts)
    total_freq = sum(freq for _, freq in top_5_terms)

    now = datetime.now()
    day_str = now.strftime("%d.%m.%Y")
    time_slot_str = get_current_time_slot(now.hour)

    csv_file = "table1_monitoring.csv"
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode="a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["День", "Час", "Топ 5", "Частота", "Сума частот", "Коментар"])

        for i, (term, freq) in enumerate(top_5_terms):
            comment = "BBC, NYT" if i == 0 else ""  # Коментар лише в першому рядку блоку
            writer.writerow([day_str, time_slot_str, term, freq, total_freq, comment])

if __name__ == "__main__":
    main()