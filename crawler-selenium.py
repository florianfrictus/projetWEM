import re
import json

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from bs4 import BeautifulSoup as bs

from elasticsearch_dsl.connections import connections

from models import Game, Comment


class JVCCrawler:
    DOMAIN = 'https://www.jeuxvideo.com'

    def __init__(self, driver, n_games) -> None:
        self.driver = driver
        self.n_games = n_games
        self.counter_game = 0
        self.page_counter = 1

    def start_crawl(self, url):
        self.driver.get(url)
        html = bs(self.driver.page_source.encode('utf-8').strip(), 'lxml')

        game_urls = [a['href'] for a in html.find_all('a', class_='gameTitleLink__196nPy')]

        # take n games to get exactly n_games
        game_urls = game_urls[:len(game_urls) if self.counter_game + len(
            game_urls) <= self.n_games else self.n_games - self.counter_game]
        print(game_urls)
        for url in game_urls:
            self.driver.get(f'{JVCCrawler.DOMAIN}{url}')
            game_html = bs(self.driver.page_source.encode('utf-8').strip(), 'lxml')
            self.parse_game(game_html)

        if self.counter_game < self.n_games:
            self.page_counter += 1
            self.start_crawl(f'{JVCCrawler.DOMAIN}/tous-les-jeux/?p={self.page_counter}')

    def parse_game(self, html):
        self.counter_game += 1
        data = json.loads(
            self.driver.find_element(by=By.XPATH, value='//script[@type="application/ld+json"]').get_attribute('text'))
        name = data['name']
        genres = data['genre']
        # platform = data["gamePlatform"]
        synopsis = html.find('p', class_='gameCharacteristicsMain__synopsis').text
        release_date = html.find('div', class_='gameCharacteristicsMain__releaseDate').text.split(':')[1].strip()
        grade_users = html.find('div', class_='gameCharacteristicsMain__reviewContainer--userOpinion').find('text',
                                                                                                            class_='gameCharacteristicsMain__gaugeText').text
        try:
            grade_editoral = int(data['review']['reviewRating']['ratingValue'])
        except:
            grade_editoral = -1

        try:
            grade_users = float(grade_users)
        except (ValueError, TypeError):
            grade_users = -1.0
        print(data)
        print(f'{name} | {genres} | {release_date} | edit:{grade_editoral} | users:{grade_users}')
        print(synopsis, end='\n\n')

        # TODO: Here create Game for elasticsearch for each platform (might change the attributes for each platform)
        # TODO: pass the urls for each platforms (urls with comments)
        # TODO: Change parameters in parse_comments to pass the game init

        first_comment_url = html.find("div", 'gameCharacteristicsMain__reviewContainer--userOpinion').find("a")["href"]
        self.driver.get(f"{JVCCrawler.DOMAIN}{first_comment_url}")
        comment_html = bs(self.driver.page_source.encode('utf-8').strip(), 'lxml')
        comments_url = [link["href"] for link in comment_html.find_all("a", 'gameHeaderBanner__platformLink')]
        self.parse_comments(first_comment_url, firsturl=True)
        for comment_url in comments_url:
            self.parse_comments(f"{JVCCrawler.DOMAIN}{comment_url}")

    def parse_comments(self, url, firsturl=False):
        if not firsturl:
            print(url)
            self.driver.get(url)
        html = bs(self.driver.page_source.encode('utf-8').strip(), 'lxml')
        reviews = html.find('div', class_='bloc-avis-tous').find_all('div', class_='bloc-avis')

        for review in reviews:
            grade = int(re.findall('\d+', review.find(class_='note-avis').text)[0])
            comment = review.find(class_='txt-avis').text.strip()
            date = review.find(class_='bloc-date-avis').text.strip()
            username = review.find(class_='bloc-pseudo-avis').text.strip()

            print(f"{username} | {date} | {grade}")
            print(comment, end='\n\n\n')

            # TODO: get the game in parameter and : game.add_comment(username, grade, comment, date)

        next_page = html.find('a', class_='pagi-suivant-actif')
        if next_page:
            next_page_href = next_page['href']
            self.parse_comments(f'{JVCCrawler.DOMAIN}{next_page_href}')


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
N_GAMES = 1

if __name__ == "__main__":
    # init_elasticsearch()
    jvc_crawler = JVCCrawler(driver, N_GAMES)
    jvc_crawler.start_crawl(URL_ALL_GAMES)

    print(f'Crawled data of {jvc_crawler.counter_game} games.')
