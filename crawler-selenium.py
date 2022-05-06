from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from selenium import webdriver

from bs4 import BeautifulSoup as bs
import json

from elasticsearch_dsl.connections import connections

from models import Game, Comment

op = webdriver.ChromeOptions()
op.add_argument('--headless')
op.add_argument("--log-level=3")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=op)

# --- CONSTANTS ---
DOMAIN = 'https://www.jeuxvideo.com'
URL_ALL_GAMES = f'{DOMAIN}/tous-les-jeux/'
N_GAMES = 30

counter_game = 0
page_counter = 1

def start_crawl(url):
    global page_counter

    driver.get(url)
    html = bs(driver.page_source.encode('utf-8').strip(), 'lxml')

    game_urls = [a['href'] for a in html.find_all('a',class_='gameTitleLink__196nPy')]

    print(game_urls)
    for url in game_urls:
        driver.get(f'{DOMAIN}{url}')
        game_html = bs(driver.page_source.encode('utf-8').strip(), 'lxml')
        parse_game(game_html)

    if counter_game < N_GAMES:
        page_counter+=1
        start_crawl(f'{DOMAIN}/tous-les-jeux/?p={page_counter}')



def parse_game(html):
    global counter_game
    counter_game += 1
    data = json.loads(driver.find_element(by=By.XPATH,value='//script[@type="application/ld+json"]').get_attribute('text'))
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


if __name__ == "__main__":
    #init_elasticsearch()
    start_crawl(URL_ALL_GAMES)


