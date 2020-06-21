from Crawler.Selenium import Selenium
from bs4 import BeautifulSoup
from urllib.request import urlretrieve
from selenium.webdriver.common.keys import Keys
import time

class GoogleImgCrawler:

    selenium = Selenium()  # 셀레니움 기능 모음


    def __init__(self, URL, photoCount, search):
        self.selenium.startSelenium(URL)



        '''
        스크롤
        for _ in range(500):
            # 가로 = 0, 세로 = 10000 픽셀 스크롤한다.
            self.selenium.driver.execute_script("window.scrollBy(0,10000)")
        '''

        # 이미지 무한 스크롤
        breaker = True
        while breaker:

            self.selenium.driver.find_element_by_xpath('//body').send_keys(Keys.CONTROL + Keys.END)
            time.sleep(0.5)

            try:
                self.selenium.driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div/div[5]/input')
                self.selenium.buttonClick('//*[@id="islmp"]/div/div/div/div/div[5]/input')
            except:

                if self.selenium.driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div/div[4]/div[2]/div[1]/div').text == '더 이상 표시할 콘텐츠가 없습니다.':
                    breaker = False
                else:
                    self.selenium.driver.find_element_by_xpath('//body').send_keys(Keys.CONTROL + Keys.END)
                    time.sleep(0.5)

        minusVal = 2

        # 이미지 클릭
        for c in range(106, photoCount+1):

            self.selenium.buttonClick('//*[@id="islrg"]/div[1]/div['+ str(c) +']')

            time.sleep(1)
            html = self.selenium.driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            img = soup.select('.n3VNCb')
            try:
                imgLST = str(img[0]).split()
                imgSRC = ''
                print(imgLST)
                for e in imgLST:
                    if e[:3] == 'src':
                        imgSRC = e
                        break

                src = imgSRC[5:-1]
                urlretrieve(src, search+'/female'+str(c-minusVal)+'.png')
            except:
                minusVal += 1




        #self.selenium.buttonClick('//*[@id="islrg"]/div[1]/div[1]')
    #def findImg(self):
        #for x in self.selenium.driver.find_elements_by_xpath('//div[contains(@class,"rg_meta")]'):


