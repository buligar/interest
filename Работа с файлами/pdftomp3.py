from gtts import gTTS
import pdfplumber
from art import tprint
from pathlib import Path

def pdf_to_mp3(file_path='test.pdf',language='en'):
    if Path(file_path).is_file() and Path(file_path).suffix == '.pdf':
        # return 'File exists!'
        print(f'[+] Original file: {Path(file_path).name}')
        print('[+] Processing...')
        with pdfplumber.PDF(open(file=file_path,mode='rb')) as pdf:
            pages = [page.extract_text() for page in pdf.pages] # пробегаемся по страницам и извлекаем текст

        text =''.join(pages) # склеиваем текст
        text = text.replace('\n','') # удаляем переносы строки

        my_audio = gTTS(text=text, lang=language, slow=False) # формируем mp3
        file_name = Path(file_path).stem # получаем имя файла
        my_audio.save(f'{file_name}.mp3') # сохраняем mp3
        return f'[+] {file_name}.mp3 saved successfully'
    else:
        return 'File not exists, check the file path'


def main():
    tprint('PDF>>TO>>MP3',font='bulbhead')
    file_path = input("\n Enter file's path")
    language = input("Choose language, for example 'en' or 'ru': ")
    print(pdf_to_mp3(file_path=file_path,language=language))

if __name__ == '__main__':
    main()