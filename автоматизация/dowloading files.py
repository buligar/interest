from icrawler.builtin import GoogleImageCrawler


def google_img_dowloader():
    crawler = GoogleImageCrawler(storage={'root_dir': './img'})
    filters = dict(
        type='photo',
        # color = 'blackandwhite',
        size='large',
        # license='noncommercial',
        # date=((2020, 1, 1), (2022, 5, 14))
    )
    crawler.crawl(keyword='blood cell',
                  max_num=5,
                  min_size=(1920,1080),
                  overwrite=True,  # перезапись файлов в папке
                  filters=filters,
                  # file_idx_offset='auto' # имена файлов начинаются с 5
                  )

def main():
    google_img_dowloader()


if __name__ == '__main__':
    main()
