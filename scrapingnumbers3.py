from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager  # ✅ 追加
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import csv
from datetime import datetime

# === 設定 ===
chrome_options = Options()
chrome_options.add_argument("--headless")  # ヘッドレスモード（画面非表示）
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
url = "https://www.mizuhobank.co.jp/takarakuji/check/numbers/numbers3/index.html"
driver.get(url)
wait = WebDriverWait(driver, 10)

data = []

try:
    # 複数件取得
    dates = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "js-lottery-date-pc")))
    numbers = driver.find_elements(By.CLASS_NAME, "js-lottery-number-pc")
    issues = driver.find_elements(By.CLASS_NAME, "js-lottery-issue-pc")
    prize_elems = driver.find_elements(By.CSS_SELECTOR, "tr.js-lottery-prize-pc strong.section__text--bold")

    num_results = min(len(dates), len(numbers), len(issues))

    for i in range(num_results):
        try:
            # 各回の要素を毎回再取得
            dates = driver.find_elements(By.CLASS_NAME, "js-lottery-date-pc")
            numbers = driver.find_elements(By.CLASS_NAME, "js-lottery-number-pc")
            issues = driver.find_elements(By.CLASS_NAME, "js-lottery-issue-pc")
            prize_elems = driver.find_elements(By.CSS_SELECTOR, "tr.js-lottery-prize-pc strong.section__text--bold")

            draw_date = datetime.strptime(dates[i].text.strip(), "%Y年%m月%d日").strftime("%Y-%m-%d")
            draw_number = issues[i].text.strip()
            main_number = str([int(d) for d in numbers[i].text.strip()])  # [6,1,1]形式

            base_index = i * 5
            def get_prize(j):
                try:
                    return int(prize_elems[base_index + j].text.replace(",", "").replace("円", "").strip())
                except:
                    return None

            data.append({
                "回別": draw_number,
                "抽せん日": draw_date,
                "本数字": main_number,
                "ストレート": get_prize(0),
                "ボックス": get_prize(1),
                "セット(ストレート)": get_prize(2),
                "セット(ボックス)": get_prize(3),
                "ミニ": get_prize(4),
            })

        except Exception as e:
            print(f"[WARNING] 回 {i+1} でエラー: {e}")

finally:
    driver.quit()

# === 保存処理 ===
csv_path = r"C:\Users\lsaka\OneDrive\デスクトップ\num3_v4\num3_v4\numbers3.csv"
try:
    existing = pd.read_csv(csv_path)
    existing_dates = existing["抽せん日"].tolist()
    fieldnames = existing.columns.tolist()
except FileNotFoundError:
    existing = pd.DataFrame()
    existing_dates = []
    fieldnames = ["抽せん日", "本数字", "回別", "ストレート", "ボックス", "セット(ストレート)", "セット(ボックス)", "ミニ"]

new_rows = [row for row in data if row["抽せん日"] not in existing_dates]

if new_rows:
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if existing.empty:
            writer.writeheader()
        writer.writerows(new_rows)

# 並び替え（日付順）
if new_rows:
    df = pd.read_csv(csv_path)
    df.sort_values("抽せん日", inplace=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[INFO] {len(new_rows)}件を保存し、日付順に並び替えました。")

# 結果表示
for row in new_rows:
    print(row)
