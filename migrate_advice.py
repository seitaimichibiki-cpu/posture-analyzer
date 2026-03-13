from app import app, db
from sqlalchemy import text
import sys
import os

# カレントディレクトリをパスに追加
sys.path.append(os.getcwd())

def migrate():
    with app.app_context():
        print("Starting migration...")
        try:
            # adviceカラムが既に存在するか確認（エラー回避のため）
            # ALTER TABLE は失敗しても例外をキャッチして続行
            db.session.execute(text("ALTER TABLE analysis_record ADD COLUMN advice TEXT"))
            db.session.commit()
            print("Successfully added 'advice' column to analysis_record table.")
        except Exception as e:
            db.session.rollback()
            print(f"Notice: Advice column might already exist or other info: {e}")
            
        print("Migration process finished.")

if __name__ == '__main__':
    migrate()
