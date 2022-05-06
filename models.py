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