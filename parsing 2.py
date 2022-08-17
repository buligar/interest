import requests
from lxml import etree
import lxml.html
import csv
import pandas as pd

def parse(url):
    try:
        api = requests.get(url)
    except:
        return
    tree = lxml.html.document_fromstring(api.text)
    text_original = tree.xpath('//*[@id="click_area"]/div//*[@class="original"]/text()')
    text_translate = tree.xpath('//*[@id="click_area"]/div//*[@class="translate"]/text()')
    with open("text.csv","w",newline='') as csv_file:
        write = csv.writer(csv_file)
        for i in range(len(text_original)):
            write.writerow(str(text_original[i]).replace('\n',''))
            write.writerow(str(text_translate[i]).replace('\n',''))
            print(text_original[i])
            print(text_translate[i])
def main():
    parse("https://www.amalgama-lab.com/songs/b/billie_eilish/lovely.html")

if __name__ == '__main__':
    main()