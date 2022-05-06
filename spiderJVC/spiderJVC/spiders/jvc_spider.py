import json
from elasticsearch_dsl.connections import connections
import scrapy
import scrapy_splash
from elasticsearch_dsl import Document, Text, Float, Integer, Date, Nested, InnerDoc


class Comment(InnerDoc):
    grade=Integer()
    comment=Text()
    date=Date()
    username=Text()

    class Index:
        name = 'comment'

class Game(Document):
    name = Text()
    editorial_grade = Integer()
    users_grade = Float()
    release_date = Date()
    synopsis = Text()
    genres = Text()
    platform = Text()

    #hasMany field
    comments = Nested(Comment)

    class Index:
        name = 'game'

    def add_comment(self, username, grade, comment, date):
        self.comments.append(Comment(username=username,grade=grade,comment=comment,date=date))

    def save(self, **kwargs):
        return super(Game, self).save(**kwargs)


class JVCGameSpider:
    NAME_CLASS = 'div.gameHeaderBanner__container span.gameHeaderBanner__title::text'
    GRADE_EDITORIAL_CLASS = 'div.gameCharacteristicsMain__reviewContainer--editorialReview text.gameCharacteristicsMain__gaugeText::text'
    GRADE_USERS_CLASS = 'div.gameCharacteristicsMain__reviewContainer--userOpinion text.gameCharacteristicsMain__gaugeText::text'
    RELEASE_DATE_CLASS = 'div.gameCharacteristicsMain__releaseDate'
    SYNOPSIS_CLASS = 'p.gameCharacteristicsMain__synopsis::text'
    ALL_GAME_HREF = 'a.gameTitleLink__196nPy::attr(href)'
    BLOCK_LINKS_PAGE_CLASS = 'div.pagination__oJAlxz a::text'


class JVCSpider(scrapy.Spider):
    name = "JVC"
    max_page = 20
    current_page = 1
    url = 'https://www.jeuxvideo.com/tous-les-jeux/'
    hard_max_page = max_page

    def start_requests(self):
        # Init connection ElasticSearch
        connections.create_connection(hosts=['localhost'])
        Game.init()

        yield scrapy_splash.SplashRequest(url=self.url, callback=self.parse, args={'wait': 0.5})

    def parse(self, response):
        print(response.url)
        data_url = response.css(JVCGameSpider.ALL_GAME_HREF).getall()

        for url in data_url:
            next_page = response.urljoin(url)
            yield scrapy.Request(next_page, callback=self.parse_game)
        if self.current_page == 1:
            self.hard_max_page = int(response.css(JVCGameSpider.BLOCK_LINKS_PAGE_CLASS).getall()[-1])
        self.current_page += 1
        if self.current_page <= self.hard_max_page and self.current_page <= self.max_page:
            yield scrapy_splash.SplashRequest(response.urljoin(f"{self.url}?p={self.current_page}"),
                                              callback=self.parse, args={'wait': 0.5})

    def parse_game(self, response):
        data = json.loads(response.xpath('//script[@type="application/ld+json"]//text()').extract_first())
        name = data['name']
        try:
            grade_editoral = data['review']['reviewRating']['ratingValue']
        except:
            grade_editoral = response.css(JVCGameSpider.GRADE_EDITORIAL_CLASS).get()
        grade_users = response.css(JVCGameSpider.GRADE_USERS_CLASS).get()
        synopsis = response.css(JVCGameSpider.SYNOPSIS_CLASS).get()
        genres = data['genre']
        release_date = \
        response.css(JVCGameSpider.RELEASE_DATE_CLASS).get().partition('</span>')[-1].partition('</div>')[0].strip(
            '\n').strip()
        platform = data["gamePlatform"]

        game = Game()
        game.name = name
        try:
            game.editorial_grade = int(grade_editoral)
        except ValueError:
            game.editorial_grade = -1

        try:
            game.users_grade = float(grade_users)
        except (ValueError, TypeError):
            game.users_grade = -1.0
        game.release_Date = release_date
        game.synopsis = synopsis
        game.genres = genres
        game.platform = platform

        game.save()

        print(f'---------------{name}:{grade_editoral}|{grade_users}|{release_date}|{genres}|{platform}')
