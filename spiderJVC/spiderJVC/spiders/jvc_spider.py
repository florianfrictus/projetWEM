import json
from elasticsearch_dsl.connections import connections
import scrapy
import scrapy_splash
from bs4 import BeautifulSoup as bs
import re
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

class JVCCommentSpider:
    REVIEWS_CLASS = 'div.bloc-avis-tous div.bloc-avis'
    REVIEW_GRADE_CLASS = 'note-avis'
    REVIEW_TXT_CLASS = 'txt-avis'
    REVIEW_DATE_CLASS = 'bloc-date-avis'
    REVIEW_USERNAME_CLASS = 'bloc-pseudo-avis'
    PAGES_CLASS = 'div.bloc-liste-num-page'

class JVCSpider2(scrapy.Spider):
    name="JVC-comment"
    url="https://www.jeuxvideo.com/jeux/pc/jeu-1056360/avis/"

    def start_requests(self):
        connections.create_connection(hosts=['localhost'])
        self.script3 = '''
        function main(splash, args)
            local ok, result = splash:with_timeout(function()
            --enabling the return of splash response
            splash.request_body_enabled = true
            --set your user agent
            splash:set_user_agent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36')
            splash.private_mode_enabled = false
            splash.indexeddb_enabled = true
            splash:set_viewport_full()
            splash:send_keys("<Tab>")
            splash:mouse_hover(100,100)
            splash.scroll_position = {y=200}
            --visit the given url
            local url = args.url
            local ok, reason = splash:go(url)
            if ok then
                --if no error found, wait for 1 second for the page to render
                splash:wait(1)
                --store the html content in a variable
                local content = assert(splash:html())
                --return the content
                return content
            end
        end,60)

        return result

        end
        '''
        #yield scrapy_splash.SplashRequest(url=self.url, callback=self.parse, args={'wait': 5})
        yield scrapy_splash.SplashRequest(url=self.url, callback=self.parse, endpoint='execute',args={'http_method': 'POST','url': self.url,'lua_source':self.script3,'timeout':60})

    def parse(self, response):
        #data = json.loads(response.xpath('//script[@type="application/ld+json"]//text()').extract_first())
        print(response)
        print(response.css("li.gameHeaderBanner__platform").getall())
        self.parse_page_comment(response)

        
        #print(reviews)

    def parse_page_comment(self, response):
        reviews = response.css(JVCCommentSpider.REVIEWS_CLASS).getall()

        print(f"There's {len(reviews)} reviews in {response.url}")
        for review in reviews:
            review_html = bs(review,"html.parser")
            grade =int(re.findall('\d+',review_html.find(class_=JVCCommentSpider.REVIEW_GRADE_CLASS).text)[0])
            comment = review_html.find(class_=JVCCommentSpider.REVIEW_TXT_CLASS).text.strip()
            date = review_html.find(class_=JVCCommentSpider.REVIEW_DATE_CLASS).text.strip()
            username = review_html.find(class_=JVCCommentSpider.REVIEW_USERNAME_CLASS).text.strip()

            #print(f"{username} | {date} | {grade}")
            #print(comment,end='\n\n\n')

            #TODO: get the game in parameter and : game.add_comment(username, grade, comment, date)
            #comments.append(Comment(username=username,grade=grade,comment=comment,date=date))

        # Check if there's another page
        pages = response.css(JVCCommentSpider.PAGES_CLASS).get()
        print(pages)
        pages = bs(pages,"html.parser")

        
            

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
