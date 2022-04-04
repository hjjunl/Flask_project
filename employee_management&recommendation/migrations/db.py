from flask_script import Manager
from flask_migrate import Migrate
from config import SQLALCHEMY_DATABASE_URI
from app import app, db

migrate = Migrate(app, db)

manager = Manager(app)
manager.add_command('db', Migrate)

if __name__ == '__main__':
    manager.run()