import os
import requests
import csv
import json
from datetime import datetime


def collect_data():
    t_date = datetime.now().strftime('%d_%m_%Y')
    proxies = {
        'https': f'http://{os.getenv("LOGIN")}:{os.getenv("PASSWORD")}@85.143.112.90' # Твой IP
    }
    response = requests.get(url='https://www.lifetime.plus/api/analysis2',proxies=proxies)
    categories = response.json()['categories']

    # Создание json
    with open(f'info_{t_date}.json','w') as file:
        json.dump(response.json(),file,indent=4,ensure_ascii=False)

    # Чтение json
    with open('info_06_06_2022.json') as f:
        file_content = f.read()
        templates = json.loads(file_content)
    categories = templates['categories']

    # парсинг json
    result = []
    for c in categories:
        c_name = c.get('name').strip()
        c_items = c.get('items')
        for item in c_items:
            item_name = item.get('name').strip()
            item_price = item.get('price')
            item_desc = item.get('description').strip()

            if 'β' in item_desc:
                item_desc = item_desc.replace('β', 'B')
            if 'γ' in item_desc:
                item_desc = item_desc.replace('γ', 'Y')
            item_wt = item.get('days')
            item_bio = item.get('biomaterial')

            result.append(
                [c_name, item_name, item_bio, item_desc, item_price, item_wt]
            )
    with open(f'1result_{t_date}.csv', 'a') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(
            (
                'Категория',
                'Анализ',
                'Биоматериал',
                'Описание',
                'Стоимость',
                'Готовность дней'
            )
        )
        writer.writerows(result)
def main():
    collect_data()

if __name__ == '__main__':
    main()