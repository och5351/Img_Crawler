from CrawlerController import CrawlerController
import os




if __name__ == '__main__':
    search = '남자 증명사진'
    URL = 'https://www.google.co.kr/search?hl=ko&tbm=isch&sxsrf=ALeKk02X3ldkQLXoU4MNT37hIVzogcVXKA%3A1592547520622&source=hp&biw=1920&bih=937&ei=wFjsXrfdI8GXr7wPn5qssAk&q='+ search +'&oq='+search+'&gs_lcp=CgNpbWcQAzICCAAyAggAMgIIADICCAAyAggAMgIIADICCAAyAggAMgIIADoHCCMQ6gIQJzoECCMQJzoFCAAQsQM6BQgAEIMBUI4wWLS5BGCPvARoDXAAeAOAAY4BiAG8FJIBBDQuMjGYAQCgAQGqAQtnd3Mtd2l6LWltZ7ABCg&sclient=img&ved=0ahUKEwi3vYLqnY3qAhXBy4sBHR8NC5YQ4dUDCAc&uact=5'
    photoCount = 500

    if not os.path.exists(search):
        os.mkdir(search)

    CrawlerController(URL, photoCount, search)