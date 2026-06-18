import scrapy
from urllib.parse import urljoin, urlparse
from pathlib import Path
import re
import enum
import pandas as pd
import os
from datetime import datetime
from datetime import timedelta
import io
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text
import traceback


@enum.unique
class DocType(enum.Enum):
    CHARTER = "Charter"
    APPLICATION = "Application"
    RENEWAL = "Renewal"
    APPROVAL = "Approval"
    MINUTES = "Minutes"
    LETTER_OF_INTENT = "Letter of Intent"
    MODIFICATION_APPLICATION = "Modification Application"
    CHARTER_RENEWAL = "Charter Renewal"
    CHARTER_PROPOSAL = "Charter Proposal"


# Keywords for regex match (case-insensitive)
# Map keywords in filenames to document type
# This is in order of guessing priority
KEYWORDS = {
    "enrollment cap": DocType.APPLICATION,
    "letter of intent": DocType.LETTER_OF_INTENT,
    "refinance": DocType.MODIFICATION_APPLICATION,
    "minutes": DocType.MINUTES,
    "expansion": DocType.MODIFICATION_APPLICATION,
    "startup": DocType.APPLICATION,
    "new school": DocType.APPLICATION,
    "merger": DocType.MODIFICATION_APPLICATION,
    "surrender": DocType.MODIFICATION_APPLICATION,
    "charter proposal": DocType.CHARTER,
    # "charter": DocType.CHARTER,
    "application": DocType.APPLICATION,
    "renewal": DocType.RENEWAL,
    "approval": DocType.APPROVAL,
    "modification application": DocType.MODIFICATION_APPLICATION,
}

class DocSpider(scrapy.Spider):
    context_window = 300
    name = "doc_spider"
    doc_regex = re.compile(r"\.(pdf|docx?|txt)$", re.IGNORECASE)

    custom_settings = {
        # "CONCURRENT_REQUESTS": 32,
        # "LOG_LEVEL": "INFO",
        "JOBDIR": "crawl_state",  # Allows resuming
        "FEEDS": {
            "visited_urls.json": {"format": "json", "overwrite": False, "indent": 4},
            # "keywords.csv": {"format": "csv", "overwrite": False},
        },
        "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/115.0.0.0 Safari/537.36",
        "DEFAULT_REQUEST_HEADERS": {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language":  "en-US,en;q=0.9",
        },
    }

    def __init__(self, sites_csv, visited_path="visited_urls.json", resume=True, *args, **kwargs):
        """
        sites_csv: INPUT, csv with columns name and url
        visited_path: OUTPUT, json with columns url, last_modified, name, last_visited
        resume: whether to resume from the last state
        """
        super().__init__(*args, **kwargs)
        self.visited_path = visited_path
        self.sites_csv = sites_csv

        urls = pd.read_csv(self.sites_csv)
        self.start_urls = urls['url'].tolist()
        self.names = urls['name'].tolist()

        self.visited = set()
        if resume and os.path.exists(self.visited_path):
            self.visited = set(pd.read_json(self.visited_path)["url"].tolist())
            # If we can't parse the file, abort!!
            if not self.visited:
                self.logger.error(f"Failed to parse {self.visited_path}")
                raise Exception(f"Failed to parse {self.visited_path}")

        self.logger.info(f"Resuming from {len(self.visited)} visited urls")

        # ensure the error log file exists:
        Path("failed_urls.txt").touch(exist_ok=True)

        self.logger.setLevel("INFO")

    def parse(self, response):
        try:
            # Spider automatically skips visited urls, so this really only runs if we resume but don't have JOBDIR set
            if response.url in self.visited:
                self.logger.info(f"SKIPPED - Skipping {response.url} because it's already visited")
                return

            last_modified = response.headers.get("Last-Modified")
            if last_modified:
                last_modified = last_modified.decode("utf-8")
                last_modified = datetime.strptime(last_modified, "%a, %d %b %Y %H:%M:%S %Z")

            try:
                name = self.names[self.start_urls.index(response.url)]
            except ValueError:
                name = None

            rtn = {
                "url": response.url,
                "last_modified": last_modified,
                "name": name,
                "last_visited": datetime.now(),
                "type": None,
                "extension": None,
                "keyword_matches": [],
                "total_length": None,
                "match_data": [],
                "downloads": [],
                "is_document": bool(re.search(self.doc_regex, response.url)),
                "skipped": False,
                "status": "",
            }

            # if it's older than 6 months, skip it, but still traverse sub links, cause they might be newer
            if last_modified and last_modified < datetime.now() - timedelta(days=6*30):
                self.logger.info(f"Skipping {response.url} because it's older than 6 months")
                rtn['status'] = "skipped: old"
                rtn['skipped'] = True
            else:
                # Now actually deal with the page

                # Check if the page is a document
                if rtn['is_document']:
                    self.logger.info(f"Found document {response.url}")
                    main_text = self.extract_pdf_text(response)
                    rtn['type'] = self.guess_doc_type(response.url)
                    rtn['extension'] = response.url.split(".")[-1]
                else:
                    self.logger.info(f"Found page {response.url}")
                    main_text = self.extract_main_text(response)
                    rtn['downloads'] = self.get_page_downloads(response)

                # So we can extract a ratio of how much of the document in post-processing
                rtn['total_length'] = len(main_text)

                # If it's a document or not, check for keywords either way
                for kw in KEYWORDS:
                    for m in re.finditer(kw, main_text, re.IGNORECASE):
                        rtn['keyword_matches'].append(kw)
                        rtn['match_data'].append({
                            'keyword': kw,
                            'text': m.group(),
                            'start': m.start(),
                            'end': m.end(),
                            'context': main_text[m.start()-self.context_window:m.end()+self.context_window],
                            })
                    # Also search the URL itself for keywords
                    if (m := re.search(kw, response.url, re.IGNORECASE)):
                        rtn['keyword_matches'].append(kw)
                        rtn['match_data'].append({
                            'keyword': kw,
                            'text': m.group(),
                            'start': m.start(),
                            'end': m.end(),
                            'context': 'URL',
                            })

                if rtn['keyword_matches']:
                    rtn['status'] = "keyword match"

            yield rtn

            if not rtn['is_document']:
                for href in response.css("a::attr(href)").getall():
                    # self.logger.info(f"Found link {href}")
                    # abs_url = urljoin(response.url, href)
                    # If it's a document, and we request it, it'll just go to the if statement above
                    if (
                        self.is_same_domain(response.url, href) and
                        not href.startswith('javascript:') and
                        not href in self.visited
                    ):
                        yield scrapy.Request(self.normalize_url(href, response), callback=self.parse)

        except Exception as e:
            self.logger.error(f"Failed to parse {response.url}: {e}")
            # Log to a different file
            with open("failed_urls.txt", "a") as f:
                f.write(f"{response.url}\n{str(e)}\n{traceback.format_exc()}\n\n\n")
            return

    def normalize_url(self, url, response):
        return url if url.startswith("http") else urljoin(response.url, url)

    def extract_pdf_text(self, response):
        """
        Given a Scrapy Response to a PDF URL,
        return all extracted text as a string using PyPDF2.
        """
        def extract_pdfminer(pdf_bytes):
            try:
                text = extract_text(pdf_bytes)
            except Exception as e:
                self.logger.warning(f"PDF parse using pdfminer failed at {response.url}: {e}")
            return text.strip()

        def extract_pdf2(pdf_bytes):
            text = ""
            try:
                reader = PdfReader(pdf_bytes)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            except Exception as e:
                self.logger.warning(f"PDF parse using PyPDF2 failed at {response.url}: {e}")
            return text.strip()

        pdf_bytes = io.BytesIO(response.body)
        try:
            # Try PyPDF2 first, cause it uses C bindings and is faster
            return extract_pdf2(pdf_bytes)
        except:
            return extract_pdfminer(pdf_bytes)

    def get_page_downloads(self, response, scan_js=False):
        """
        Extract likely download URLs from a page.
        Returns a list of absolute URLs.
        """
        downloads = []

        # 1. Look for <a href="..."> with common file extensions
        exts = (".pdf", ".doc", ".docx", ".xls", ".xlsx", ".zip", ".rar", ".7z", ".csv")
        for href in response.css("a::attr(href)").getall():
            if any(href.lower().endswith(ext) for ext in exts):
                downloads.append(urljoin(response.url, href))

        # 2. Look for <form action="..."> (might trigger a file response)
        for action in response.css("form::attr(action)").getall():
            downloads.append(urljoin(response.url, action))

        # 3. Look for JS-included URLs (naive regex scan for .pdf etc.)
        if scan_js:
            text = response.text
            for ext in exts:
                idx = 0
                while True:
                    idx = text.lower().find(ext, idx)
                    if idx == -1:
                        break
                    # backtrack a bit to find the start of the URL
                    start = max(text.rfind('"', 0, idx), text.rfind("'", 0, idx))
                    if start != -1:
                        url = text[start+1:idx+len(ext)]
                        downloads.append(urljoin(response.url, url))
                    idx += len(ext)

        # Deduplicate, but maintain order
        return list(dict.fromkeys(downloads))

    def extract_main_text(self, response):
        for selector in ["header", "nav", "footer"]:
            for element in response.css(selector):
                element.extract()
        text = " ".join(response.css("body *::text").getall())
        return re.sub(r"\s+", " ", text).strip()

    def guess_doc_type(self, url):
        filename = url.lower()
        for kw, dtype in KEYWORDS.items():
            if kw in filename or kw.replace(" ", "_") in filename or kw.replace(" ", "-") in filename:
                return dtype
        return None

    def is_same_domain(self, base_url, target_url):
        # If it's a relative path, it's the same domain
        if not target_url.startswith("http"):
            return True
        return urlparse(base_url).netloc == urlparse(target_url).netloc
