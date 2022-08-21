import requests


def get_info():
    response = requests.get(url="https://yobit.net/api/3/info")

    with open("info.json","w") as file:
        file.write(response.text)

    return response.text

def get_ticker(coin1="btc",coin2="usd"):
    response = requests.get(url=f"https://yobit.net/api/3/ticker/{coin1}_{coin2}?ignore_invalid=1")

    with open("ticker.json","w") as file:
        file.write(response.text)

    return response.text

def get_depth(coin1="btc",coin2="usd",limit=150):
    response = requests.get(url=f"https://yobit.net/api/3/depth/{coin1}_{coin2}?limit={limit}&ignore_invalid=1")
    with open("depth.json", "w") as file:
        file.write(response.text)

    bids = response.json()[f"{coin1}_usd"]["bids"]

    total_bids_amount = 0
    for item in bids:
        price = item[0]
        coin_amount = item[1]
        total_bids_amount += price * coin_amount
    return f"Total bids: {total_bids_amount} $"

def get_trades(coin1="btc",coin2="usd",limit=150):
    response = requests.get(url=f"https://yobit.net/api/3/trades/{coin1}_{coin2}?limit={limit}&ignore_invalid=1")

    with open("trades.json", "w") as file:
        file.write(response.text)

    total_trade_ask = 0
    total_trade_bid = 0

    for item in response.json()[f"{coin1}_{coin2}"]:
        if item["type"] == "ask":
            total_trade_ask += item["price"] * item["amount"]
        else:
            total_trade_bid += item["price"] * item["amount"]

    info = f"[-] TOTAL {coin1} SELL: {total_trade_ask:.2f} $\n[+] TOTAL {coin1} BUY: {total_trade_bid:.2f} $"

    return info
def main():
    # get_info()
    get_ticker()
    print(get_depth(coin1="doge",limit=2000))
    print(get_trades(coin1="btc"))
if __name__ == '__main__':
    main()