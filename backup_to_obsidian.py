import requests
import os
from datetime import datetime

# 設定
BACKUP_TOKEN = "seitai-backup-2026-safe"
RENDER_URL = f"https://posture-analyzer-clinic.onrender.com/admin/backup?token={BACKUP_TOKEN}"
# 2026 会員バックアップ先フォルダ（絶対パス）
BACKUP_DIR = "/Users/ishikawagai/Desktop/整体院導/姿勢AIバックアップ"
# ファイル名に日付を入れる（例: members_backup_2026-03-12.md）
FILENAME = f"members_backup_{datetime.now().strftime('%Y-%m-%d')}.md"

def download_backup():
    print(f"Connecting to {RENDER_URL}...")
    try:
        # 管理者認証が必要な場合はログインセッションが必要ですが、
        # 現状のエンドポイントが認証必須なため、ブラウザでログインした状態の
        # クッキーを指定するか、API用のトークンを検討する必要があります。
        # ここではシンプルにダウンロードを試みます。
        response = requests.get(RENDER_URL)
        
        if response.status_code == 200:
            if not os.path.exists(BACKUP_DIR):
                os.makedirs(BACKUP_DIR)
            
            filepath = os.path.join(BACKUP_DIR, FILENAME)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(response.text)
            
            print(f"Success! Backup saved to: {filepath}")
        elif response.status_code == 302 or "login" in response.url:
            print("Error: ログインが必要です。ブラウザで一度ログインしてから再度試すか、認証情報を設定してください。")
        else:
            print(f"Failed to download. Status code: {response.status_code}")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    download_backup()
