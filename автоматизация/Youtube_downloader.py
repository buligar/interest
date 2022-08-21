from pytube import YouTube

def main():
    yt = YouTube(
        "https://www.youtube.com/watch?v=u3p7XbfwdKA&ab_channel=%D0%A1%D1%82%D1%83%D0%BB%D1%8B%D1%87%D0%9F%D0%B0%D0%BF%D0%B8%D1%87%D0%B0")

    print("Title:", yt.title)
    print("Views:", yt.views)

    yd = yt.streams.get_highest_resolution()

    yd.download("./youtube")

if __name__ == "__main__":
    main()
