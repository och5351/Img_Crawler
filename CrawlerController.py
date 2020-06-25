from Crawler.GoogleImgCrawler import GoogleImgCrawler
import os

class CrawlerController:
    GIC = None  # 구글 이미지 크롤러 컨트롤러

    search = None

    def __init__(self, URL, photoCount, search):
        self.GIC = GoogleImgCrawler(URL, photoCount, search)
        #self.search = search
        #self.collecting_human()

