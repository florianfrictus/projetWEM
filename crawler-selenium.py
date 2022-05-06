from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from selenium import webdriver

from bs4 import BeautifulSoup as bs
import json

from elasticsearch_dsl.connections import connections

from models import Game, Comment


class JVCCrawler:
    DOMAIN = 'https://www.jeuxvideo.com'
    
    def __init__(self,driver,n_games) -> None:
        self.driver = driver
        self.n_games = n_games
        self.counter_game = 0
        self.page_counter = 1

    def start_crawl(self,url):
        self.driver.get(url)
        html = bs(self.driver.page_source.encode('utf-8').strip(), 'lxml')

        game_urls = [a['href'] for a in html.find_all('a',class_='gameTitleLink__196nPy')]

        print(game_urls)
        for url in game_urls:
            self.driver.get(f'{JVCCrawler.DOMAIN}{url}')
            game_html = bs(self.driver.page_source.encode('utf-8').strip(), 'lxml')
            self.parse_game(game_html)

        if self.counter_game < self.n_games:
            self.page_counter+=1
            self.start_crawl(f'{JVCCrawler.DOMAIN}/tous-les-jeux/?p={self.page_counter}')


    def parse_game(self,html):
        self.counter_game += 1
        data = json.loads(self.driver.find_element(by=By.XPATH,value='//script[@type="application/ld+json"]').get_attribute('text'))
        name = data['name']
        genres = data['genre']
        platform = data["gamePlatform"]
        synopsis = html.find('p',class_='gameCharacteristicsMain__synopsis').text
        release_date = html.find('div',class_='gameCharacteristicsMain__releaseDate').text.split(':')[1].strip()
        grade_users = html.find('div',class_='gameCharacteristicsMain__reviewContainer--userOpinion').find('text',class_='gameCharacteristicsMain__gaugeText').text
        try:
            grade_editoral = int(data['review']['reviewRating']['ratingValue'])
        except:
            grade_editoral = -1

        try:
            grade_users = float(grade_users)
        except (ValueError, TypeError):
            grade_users = -1.0

        print(f'{name} | {genres} | {platform} | {release_date} | edit:{grade_editoral} | users:{grade_users}')
        print(synopsis,end='\n\n')

def init_elasticsearch():
    connections.create_connection(hosts=['localhost'])
    Game.init()

# Init Webdriver selenium
op = webdriver.ChromeOptions()
op.add_argument('--headless')
op.add_argument("--log-level=3")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=op)

# --- CONSTANTS ---
URL_ALL_GAMES = f'{JVCCrawler.DOMAIN}/tous-les-jeux/'
N_GAMES = 30

if __name__ == "__main__":
    #init_elasticsearch()
    jvc_crawler = JVCCrawler(driver,N_GAMES)
    jvc_crawler.start_crawl(URL_ALL_GAMES)

    print(f'Crawled data of {jvc_crawler.counter_games} games.')


