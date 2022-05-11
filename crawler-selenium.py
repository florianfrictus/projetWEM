from datetime import date as dt
from datetime import timedelta as td
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
    FRENCH_MONTHS_COMS = {'janv.': 1, 'févr.': 2, 'mars': 3, 'avr.': 4, 'mai': 5, 'juin': 6, 'juil.': 7, 'août': 8,
                          'sept.': 9, 'oct.': 10, 'nov.': 11, 'déc.': 12}
    FRENCH_MONTHS_MAIN = {'janvier': 1, 'février': 2, 'mars': 3, 'avril': 4, 'mai': 5, 'juin': 6, 'juillet': 7,
                          'août': 8, 'septembre': 9, 'octobre': 10, 'novembre': 11, 'décembre': 12}

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
            try:
                self.parse_game(game_html)
            except:
                pass

        if self.counter_game < self.n_games:
            self.start_crawl(f'{JVCCrawler.DOMAIN}/tous-les-jeux/?p={self.page_counter}')

    def parse_game(self, html):
        self.counter_game += 1
        data = json.loads(
            self.driver.find_element(by=By.XPATH, value='//script[@type="application/ld+json"]').get_attribute('text'))
        name = data['name']
        genres = data['genre']
        # platform = data["gamePlatform"]
        try:
            synopsis = html.find('p', class_='gameCharacteristicsMain__synopsis').text
            release_date = html.find('div', class_='gameCharacteristicsMain__releaseDate').text.split(':')[1].strip()
        except(TypeError, AttributeError):
            synopsis = ""
            release_date = ""
        try:
            grade_users = html.find('div', class_='gameCharacteristicsMain__reviewContainer--userOpinion').find('text',
                                                                                                                class_='gameCharacteristicsMain__gaugeText').text
        except:
            grade_users = -1
        try:
            grade_editoral = int(data['review']['reviewRating']['ratingValue'])
        except:
            grade_editoral = -1

        try:
            grade_users = float(grade_users)
        except (ValueError, TypeError):
            grade_users = -1.0

        # Format release_date
        date_split = release_date.split(' ')
        try:
            release_date = dt(year=int(date_split[2]), month=JVCCrawler.FRENCH_MONTHS_COMS[date_split[1]],
                              day=int(date_split[0]))
        except (IndexError, KeyError, ValueError):
            release_date = None  # not valid or haven't released yet

        # print(data)
        # print(f'{name} | {genres} | {release_date} | edit:{grade_editoral} | users:{grade_users}')
        # print(synopsis, end='\n\n')

        if release_date:
            self.page_counter += 1
            first_comment_url = ""

            try:
                first_comment_url = html.find("div", 'gameCharacteristicsMain__reviewContainer--userOpinion').find("a")[
                    "href"]
                is_reviewed = True
            except (TypeError, AttributeError):
                is_reviewed = False

            if is_reviewed:
                self.driver.get(f"{JVCCrawler.DOMAIN}{first_comment_url}")
                comment_html = bs(self.driver.page_source.encode('utf-8').strip(), 'lxml')
                comments_url = [link["href"] for link in comment_html.find_all("a", 'gameHeaderBanner__platformLink')]

                game = self.__create_game(name, grade_editoral, grade_users, release_date, synopsis, genres)
                self.parse_comments(game, first_comment_url, firsturl=True)

                for comment_url in comments_url:
                    game = self.__create_game(name, grade_editoral, grade_users, release_date, synopsis, genres)
                    self.parse_comments(game, f"{JVCCrawler.DOMAIN}{comment_url}")

    def parse_comments(self, game, url, firsturl=False):
        if not firsturl:
            # print(url)
            self.driver.get(url)
        save_game = True
        data = json.loads(
            self.driver.find_element(by=By.XPATH, value='//script[@type="application/ld+json"]').get_attribute('text'))

        try:
            game.platform = data['itemReviewed']['gamePlatform']
        except KeyError:
            save_game = False

        if save_game:
            html = bs(self.driver.page_source.encode('utf-8').strip(), 'lxml')
            reviews = html.find('div', class_='bloc-avis-tous').find_all('div', class_='bloc-avis')

            for review in reviews:
                grade = int(re.findall('\d+', review.find(class_='note-avis').text)[0])
                comment = review.find(class_='txt-avis').text.strip()
                date = review.find(class_='bloc-date-avis').text.strip()
                username = review.find(class_='bloc-pseudo-avis').text.strip()

                # Date format
                res_today = re.findall('il y a', date)
                res_yesterday = re.findall('hier', date)
                current_date = dt.today()
                if len(res_today) > 0:
                    date = current_date
                elif len(res_yesterday) > 0:
                    yesterday = current_date - td(days=1)
                    date = yesterday
                else:
                    date_split = date.split(' ')
                    try:
                        year = int(date_split[4])
                    except ValueError:
                        year = current_date.year  # if it can't cast in int there's to year displayed and it's the current one

                    try:
                        date = dt(year=year, month=JVCCrawler.FRENCH_MONTHS_COMS[date_split[3]], day=int(date_split[2]))
                    except KeyError:
                        date = None

                # print(f"{username} | {date} | {grade}")
                # print(comment, end='\n\n\n')
                game.add_comment(username, grade, comment, date)  # TODO: parse date

            next_page = html.find('a', class_='pagi-suivant-actif')
            if next_page:
                next_page_href = next_page['href']
                self.parse_comments(game, f'{JVCCrawler.DOMAIN}{next_page_href}')
            else:
                print(f'Game {game.name}|{game.platform} saved (with {len(game.comments)} comments).')
                game.save()

    def __create_game(self, name, grade_editoral, grade_users, release_date, synopsis, genres) -> Game:
        game = Game()
        game.name = name
        game.editorial_grade = grade_editoral
        game.users_grade = grade_users
        game.release_date = release_date
        game.synopsis = synopsis
        game.genres = genres
        return game


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
N_GAMES = 1000

if __name__ == "__main__":
    init_elasticsearch()
    jvc_crawler = JVCCrawler(driver, N_GAMES)
    jvc_crawler.start_crawl(URL_ALL_GAMES)

    print(f'Crawled data of {jvc_crawler.counter_game} games.')
