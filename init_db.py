import os
import sys

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app, db, User
from werkzeug.security import generate_password_hash

def init_db():
    print("データベース初期化中...")
    with app.app_context():
        # 既存のテーブルをすべて削除して再作成（開発中のみ）
        db.drop_all()
        db.create_all()
        
        # 管理者（導様）アカウントの作成
        admin_email = os.environ.get('ADMIN_EMAIL', 'seitaimichibiki@gmail.com')
        admin_password = os.environ.get('ADMIN_PASSWORD', 'gai1124714')
        
        admin = User(
            email=admin_email,
            password=generate_password_hash(admin_password, method='pbkdf2:sha256'),
            is_active_member=True,
            is_admin=True
        )
        
        db.session.add(admin)
        db.session.commit()
        
        print(f"初期会員データを作成しました:")
        print(f"  [管理者] seitaimichibiki@gmail.com / gai1124714")

if __name__ == '__main__':
    init_db()
