import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from datetime import datetime
import os
import sys
# import camelot

# Извлечение таблиц из html-страницы
# simpsons = pd.read_html('https://ru.wikipedia.org/wiki/%D0%A1%D0%BF%D0%B8%D1%81%D0%BE%D0%BA '
#                         '%D1%8D%D0%BF%D0%B8%D0%B7%D0%BE%D0%B4%D0%BE%D0%B2_'
#                         '%D0%BC%D1%83%D0%BB%D1%8C%D1%82%D1%81%D0%B5%D1%80%D0%B8%D0%B0%D0%BB%D0%B0_'
#                         '%C2%AB%D0%A1%D0%B8%D0%BC%D0%BF%D1%81%D0%BE%D0%BD%D1%8B%C2%BB_'
#                         '(%D1%81%D0%B5%D0%B7%D0%BE%D0%BD%D1%8B_1%E2%80%9420)')
# len(simpsons)
# print(simpsons[1])

# скачать все .csv с сайта
# df_premier21 = pd.read_csv('https://www.football-data.co.uk/mmz4281/2223/E0.csv')
# df_premier21.rename(columns={'FTHG':'home_goals',
#                              'FTAG':'away_goals'}, inplace=True)
# print(df_premier21)

# Извлечение таблиц из pdf-файла
# tables = camelot.read_pdf('Б.pdf', pages='39')
# print(tables[0])
# tables.export('Review.csv', f='csv', compress=True)
# tables[0].to_csv('Review.csv')

# HTML Tags & Elements

# Tags
# <head>
# <body>
# <header>
# <article>
# <p> : paragraph
# <h1>, <h2>, <h3> : heading
# <div> : divider делитель
# <nav> : navigational
# <li> : list item
# <a> : anchor якорь
# <button>
# <table>
# <td> : table data
# <tr> : table row element
# <ul> : unordered list
# <iframe> : встроить страницу в другую страницу

# Извлечь заголовок и подзаголовок сайта
application_path = os.path.dirname(sys.executable)
now = datetime.now()
# DDMMYYYY
day_month_year = now.strftime("%d%m%Y")
website = "https://www.rbc.ru/"
path = "/home/buligar/PycharmProjects/automate/chromedriver_linux64/chromedriver"

#headless-mode
options = Options()
options.headless = True
service = Service(executable_path=path)
driver = webdriver.Chrome(service=service, options=options)
driver.get(website)
containers = driver.find_elements(by="xpath", value='//div[@class="main__inner l-col-center"]')

titles = []
subtitles = []
links = []

for container in containers:
    title = container.find_element(by="xpath", value='./div/a/span/span')
    print(title)
    titles.append(title)

my_dict = {'title': titles}
df_headlines = pd.DataFrame(my_dict)
file_name = f'headline-{day_month_year}.csv'
final_path = os.path.join(application_path, file_name)
df_headlines.to_csv(final_path, sep=";")

driver.quit()