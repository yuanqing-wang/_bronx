import pandas as pd


def run():
    df = pd.read_csv("performance.csv")
    df.to_excel("performance.xlsx")
    import os

    os.system("rclone copy performance.xlsx remote:lilac")


if __name__ == "__main__":
    run()
