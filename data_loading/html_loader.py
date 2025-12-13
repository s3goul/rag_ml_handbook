"""
Скрипт для скачивания статей из Яндекс ML Handbook.
Использует Selenium для рендеринга JavaScript-контента.
"""

import json
import os
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By


class HandbookScraper:
    """Скрапер для загрузки статей из Яндекс ML Handbook."""
    
    HANDBOOK_URL = "https://education.yandex.ru/handbook/ml"
    ARTICLE_PATTERN = "/handbook/ml/article/"
    PAGE_LOAD_DELAY = 15
    
    def __init__(self, output_dir: str = "handbook_pages"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.driver = None
        self._url_mapping = {}
    
    def _init_browser(self) -> webdriver.Chrome:
        """Инициализация браузера с нужными настройками."""
        opts = Options()
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_argument(
            "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        return webdriver.Chrome(options=opts)
    
    def _wait_for_content(self):
        """Ожидание загрузки динамического контента."""
        time.sleep(self.PAGE_LOAD_DELAY)
    
    def _collect_article_links(self) -> list[str]:
        """Сбор ссылок на все статьи с главной страницы."""
        print(f"Открываю главную страницу: {self.HANDBOOK_URL}")
        self.driver.get(self.HANDBOOK_URL)
        self._wait_for_content()
        
        all_links = self.driver.find_elements(By.TAG_NAME, "a")
        article_links = []
        
        for link in all_links:
            href = link.get_attribute("href")
            if href and self.ARTICLE_PATTERN in href:
                article_links.append(href)
        
        # Убираем дубликаты, сохраняя порядок
        seen = set()
        unique_links = []
        for url in article_links:
            if url not in seen:
                seen.add(url)
                unique_links.append(url)
        
        print(f"Найдено {len(unique_links)} статей")
        return unique_links
    
    def _download_page(self, url: str, page_idx: int) -> str:
        """Скачивание и сохранение одной страницы."""
        filename = f"page_{page_idx}.html"
        filepath = self.output_dir / filename
        
        print(f"[{page_idx}] Загружаю: {url}")
        self.driver.get(url)
        self._wait_for_content()
        
        html_content = self.driver.page_source
        filepath.write_text(html_content, encoding="utf-8")
        print(f"[{page_idx}] Сохранено: {filepath}")
        
        return str(filepath)
    
    def _save_url_mapping(self):
        """Сохранение соответствия файлов и URL."""
        # URL -> filename
        url2file = {url: fname for url, fname in self._url_mapping.items()}
        url2file_path = self.output_dir / "url2filename.json"
        url2file_path.write_text(
            json.dumps(url2file, ensure_ascii=False, indent=4),
            encoding="utf-8"
        )
        
        # filename -> URL  
        file2url = {fname: url for url, fname in self._url_mapping.items()}
        file2url_path = self.output_dir / "filename2url.json"
        file2url_path.write_text(
            json.dumps(file2url, ensure_ascii=False, indent=4),
            encoding="utf-8"
        )
        
        print(f"Маппинг сохранён в {self.output_dir}")
    
    def run(self):
        """Основной метод запуска скрапера."""
        self.driver = self._init_browser()
        
        try:
            articles = self._collect_article_links()
            
            for idx, article_url in enumerate(articles, start=1):
                saved_path = self._download_page(article_url, idx)
                self._url_mapping[article_url] = saved_path
            
            self._save_url_mapping()
            print(f"\nГотово! Скачано {len(articles)} статей.")
            
        finally:
            self.driver.quit()


if __name__ == "__main__":
    scraper = HandbookScraper(output_dir="data")
    scraper.run()
