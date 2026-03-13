import os
import sys
import uuid
import json
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for, redirect, flash
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import sqlalchemy
from sqlalchemy import text, inspect
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mail import Mail, Message
import cloudinary
import cloudinary.uploader

# 姿勢解析エンジンのインポート
from pose_analyzer import PoseAnalyzer

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'seitai-michibiki-secret-key-12345')

# DB設定: 環境変数 DATABASE_URL があれば使用（RenderのPostgres等）、なければSQLite
db_url = os.environ.get('DATABASE_URL')
if db_url:
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = db_url
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database.db')

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# バックアップ用トークンも環境変数から取得
BACKUP_TOKEN = os.environ.get('BACKUP_TOKEN', 'seitai-backup-2026-safe')

# メール設定 (Gmail SMTP)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', 'seitaimichibiki@gmail.com')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', '').strip() or None # Renderの環境変数で設定
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_USERNAME', 'seitaimichibiki@gmail.com')

# LINE Messaging API 設定
app.config['LINE_CHANNEL_ACCESS_TOKEN'] = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN', '')
app.config['LINE_CHANNEL_SECRET'] = os.environ.get('LINE_CHANNEL_SECRET', '')

# Cloudinary設定 (環境変数 CLOUDINARY_URL または個別キーを使用)
cloudinary_url_env = os.environ.get('CLOUDINARY_URL')
if cloudinary_url_env:
    cloudinary.config(cloudinary_url=cloudinary_url_env, secure=True)
else:
    cloudinary.config(
        cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
        api_key=os.environ.get('CLOUDINARY_API_KEY'),
        api_secret=os.environ.get('CLOUDINARY_API_SECRET'),
        secure=True
    )

def upload_to_cloudinary(file_path):
    """画像をCloudinaryにアップロードし、URLを返す。設定がない場合はNoneを返す。"""
    if not (os.environ.get('CLOUDINARY_URL') or os.environ.get('CLOUDINARY_CLOUD_NAME')):
        return None
    try:
        result = cloudinary.uploader.upload(file_path, folder="posture-reports")
        return result.get('secure_url')
    except Exception as e:
        print(f"Cloudinary upload failed: {e}")
        return None

mail = Mail(app)

# 起動ログ (デバッグ用)
print(f"--- 起動設定確認 ---")
print(f"MAIL_USERNAME: {app.config.get('MAIL_USERNAME')}")
pwd = app.config.get('MAIL_PASSWORD')
print(f"MAIL_PASSWORD: {'設定済み' if pwd else '未設定'}")
if pwd:
    print(f"MAIL_PASSWORD (先頭2文字): {pwd[:2]}...")
print(f"-------------------")

CORS(app)
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', code=404, title="Page Not Found"), 404

@app.errorhandler(500)
def server_error(e):
    if request.is_json or request.path.startswith('/api/') or request.path.endswith('/send'):
        return jsonify({'success': False, 'error': f'Internal Server Error: {str(e)}'}), 500
    return render_template('error.html', code=500, title="Server Error"), 500

db = SQLAlchemy(app)
app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=30)
login_manager = LoginManager(app)
login_manager.login_view = 'login' # ログインしていない場合に飛ばす先

# ─── データベース初期化・自動マイグレーション ────────────────────────────────
def init_and_migrate():
    with app.app_context():
        try:
            # テーブルの作成
            db.create_all()
            
            inspector = sqlalchemy.inspect(db.engine)
            
            # --- Userテーブルのマイグレーション ---
            user_cols = [c['name'] for c in inspector.get_columns('user')]
            with db.engine.connect() as conn:
                # パスワードリセット用
                if 'reset_token' not in user_cols:
                    conn.execute(text('ALTER TABLE "user" ADD COLUMN reset_token VARCHAR(100)'))
                    conn.execute(text('ALTER TABLE "user" ADD COLUMN reset_token_expiration DATETIME'))
                
                # ログイン制限用
                if 'failed_login_attempts' not in user_cols:
                    conn.execute(text('ALTER TABLE "user" ADD COLUMN failed_login_attempts INTEGER DEFAULT 0'))
                    conn.execute(text('ALTER TABLE "user" ADD COLUMN locked_until DATETIME'))
                
                # LINE連携用
                if 'line_access_token' not in user_cols:
                    conn.execute(text('ALTER TABLE "user" ADD COLUMN line_access_token VARCHAR(255)'))
                if 'line_channel_secret' not in user_cols:
                    conn.execute(text('ALTER TABLE "user" ADD COLUMN line_channel_secret VARCHAR(100)'))
                
                conn.commit()

            # --- AnalysisRecordテーブルのマイグレーション ---
            record_cols = [c['name'] for c in inspector.get_columns('analysis_record')]
            with db.engine.connect() as conn:
                # 顧客ID
                if 'patient_id' not in record_cols:
                    conn.execute(text('ALTER TABLE analysis_record ADD COLUMN patient_id VARCHAR(50)'))
                
                # メモ
                if 'memo' not in record_cols:
                    conn.execute(text('ALTER TABLE analysis_record ADD COLUMN memo TEXT'))
                
                # LINE送信
                if 'line_user_id' not in record_cols:
                    conn.execute(text('ALTER TABLE analysis_record ADD COLUMN line_user_id VARCHAR(100)'))
                
                # 画像ファイル名
                if 'image_filename' not in record_cols:
                    conn.execute(text('ALTER TABLE analysis_record ADD COLUMN image_filename VARCHAR(255)'))
                if 'input_filename' not in record_cols:
                    conn.execute(text('ALTER TABLE analysis_record ADD COLUMN input_filename VARCHAR(255)'))
                
                # AIアドバイス
                if 'advice' not in record_cols:
                    conn.execute(text('ALTER TABLE analysis_record ADD COLUMN advice TEXT'))
                
                conn.commit()

            # --- LineUserMappingテーブルの確認 ---
            if 'line_user_mapping' not in inspector.get_table_names():
                db.create_all()
                
            print("Database migration completed successfully.")
        except Exception as e:
            app.logger.error(f"Migration error: {e}")
            print(f"Migration error: {e}")

# ─── データベースモデル ──────────────────────────────────────────────────────
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    is_active_member = db.Column(db.Boolean, default=False) # サブスク有効フラグ
    is_admin = db.Column(db.Boolean, default=False)
    # LINE連携設定
    line_access_token = db.Column(db.String(255), nullable=True)
    line_channel_secret = db.Column(db.String(100), nullable=True)
    
    # パスワードリセット用
    reset_token = db.Column(db.String(100), unique=True, nullable=True)
    reset_token_expiration = db.Column(db.DateTime, nullable=True)
    # ログイン試行制限用
    failed_login_attempts = db.Column(db.Integer, default=0)
    locked_until = db.Column(db.DateTime, nullable=True)

class LineUserMapping(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    line_user_id = db.Column(db.String(100), nullable=False)
    display_name = db.Column(db.String(255))
    owner_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    updated_at = db.Column(db.DateTime, default=db.func.current_timestamp(), onupdate=db.func.current_timestamp())

class AnalysisLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    view_type = db.Column(db.String(20), nullable=False) # 'front', 'side', 'compare'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class AnalysisRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    patient_id = db.Column(db.String(50), nullable=True) # 顧客ID/名前
    view_type = db.Column(db.String(20)) # 'front', 'side'
    
    # 正面データ
    shoulder_angle = db.Column(db.Float)
    pelvis_angle = db.Column(db.Float)
    head_angle = db.Column(db.Float)
    ear_shift_pct = db.Column(db.Float)
    shoulder_shift_pct = db.Column(db.Float)
    pelvis_shift_pct = db.Column(db.Float)
    
    # 側面データ
    fhp_pct = db.Column(db.Float)
    rs_pct = db.Column(db.Float)
    side_pelvis_angle = db.Column(db.Float)
    trunk_pct = db.Column(db.Float)
    
    # 追記：メモ機能
    memo = db.Column(db.Text, nullable=True)
    
    # 追記：LINE連携用
    line_user_id = db.Column(db.String(100), nullable=True)
    
    # 追記：画像パス保存用
    image_filename = db.Column(db.String(255), nullable=True)
    input_filename = db.Column(db.String(255), nullable=True)
    
    # 追記：AIアドバイス
    advice = db.Column(db.Text, nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# モデル定義後にマイグレーション実行
init_and_migrate()

@app.errorhandler(500)
def handle_500_error(e):
    import traceback
    error_details = traceback.format_exc()
    app.logger.error(f"Server Error: {error_details}")
    return jsonify({
        'success': False, 
        'error': f'サーバー内部エラーが発生しました: {str(e)}',
        'details': error_details if app.debug else '管理者にお問い合わせください。'
    }), 500

@app.route('/debug/db')
@login_required
def debug_db():
    if not current_user.is_admin:
        return "Access denied", 403
    
    try:
        inspector = sqlalchemy.inspect(db.engine)
        tables = inspector.get_table_names()
        info = {}
        for table in tables:
            cols = inspector.get_columns(table)
            info[table] = [c['name'] for c in cols]
        
        return jsonify({
            'database': str(db.engine.url),
            'tables': info
        })
    except Exception as e:
        return str(e), 500

# ─── 姿勢解析設定 ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static/uploads')
MODEL_PATH = os.path.join(BASE_DIR, 'pose_landmarker_lite.task')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# エンジンの遅延初期化用
_analyzer = None

def get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = PoseAnalyzer(MODEL_PATH)
    return _analyzer

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

# ─── ルート定義 ──────────────────────────────────────────────────────────────
@app.route('/')
@login_required
def index():
    return render_template('index.html', user=current_user)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        user = User.query.filter_by(email=email).first()
        
        if user:
            # ロック状態のチェック
            if user.locked_until and user.locked_until > datetime.utcnow():
                wait_mins = int((user.locked_until - datetime.utcnow()).total_seconds() / 60) + 1
                return jsonify({'success': False, 'error': f'アカウントがロックされています。あと{wait_mins}分後に再試行してください。'}), 403

            if check_password_hash(user.password, password):
                # 成功時は失敗カウントをリセット
                user.failed_login_attempts = 0
                user.locked_until = None
                db.session.commit()
                
                # 自動ログイン設定
                remember = request.form.get('remember') == 'true'
                login_user(user, remember=remember)
                return jsonify({'success': True})
            else:
                # 失敗時はカウントアップ
                user.failed_login_attempts += 1
                if user.failed_login_attempts >= 5:
                    user.locked_until = datetime.utcnow() + timedelta(minutes=15)
                    error_msg = 'パスワードを5回間違えたため、アカウントを15分間ロックしました。'
                else:
                    remaining = 5 - user.failed_login_attempts
                    error_msg = f'パスワードが正しくありません。あと{remaining}回失敗するとロックされます。'
                
                db.session.commit()
                print(f"Login failed: Password mismatch for {email}. Attempt: {user.failed_login_attempts}")
                return jsonify({'success': False, 'error': error_msg}), 401
        else:
            print(f"Login failed: User not found: {email}")
            return jsonify({'success': False, 'error': 'メールアドレスまたはパスワードが正しくありません。'}), 401
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/api/user/settings/line', methods=['POST'])
@login_required
def update_line_settings():
    data = request.json
    current_user.line_access_token = data.get('line_access_token')
    current_user.line_channel_secret = data.get('line_channel_secret')
    db.session.commit()
    return jsonify({'success': True})

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user=current_user)

@app.route('/profile/update_password', methods=['POST'])
@login_required
def update_password():
    data = request.json
    current_pw = data.get('current_password')
    new_pw = data.get('new_password')
    
    if not check_password_hash(current_user.password, current_pw):
        return jsonify({'success': False, 'error': '現在のパスワードが正しくありません。'}), 401
    
    current_user.password = generate_password_hash(new_pw)
    db.session.commit()
    return jsonify({'success': True})

@app.route('/support')
def support():
    return render_template('support.html')

@app.route('/support/send', methods=['POST'])
def support_send():
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'データが送信されていません。'}), 400
    
    name = data.get('name')
    email = data.get('email')
    subject = data.get('subject')
    message = data.get('message')

    if not all([name, email, subject, message]):
        return jsonify({'success': False, 'error': 'すべての項目を入力してください。'}), 400

    try:
        if not app.config.get('MAIL_PASSWORD'):
            print("警告: MAIL_PASSWORD が設定されていません。")
            return jsonify({'success': False, 'error': '現在、メール送信機能が一時的に利用できません（パスワード未設定）。'}), 503

        msg = Message(
            subject=f"【お問い合わせ】{subject}",
            recipients=['seitaimichibiki@gmail.com'],
            body=f"お名前: {name}\nメールアドレス: {email}\n\n内容:\n{message}"
        )
        mail.send(msg)
        return jsonify({'success': True})
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"メール送信エラー詳細:\n{error_msg}")
        return jsonify({'success': False, 'error': f'メールの送信に失敗しました: {str(e)}'}), 500

@app.route('/stats')
@login_required
def stats():
    patient_id = request.args.get('patient_id', '')
    return render_template('stats.html', user=current_user, patient_id=patient_id)

@app.route('/api/patients/search')
@login_required
def api_search_patients():
    query = request.args.get('q', '').strip()
    # ログインユーザーに関連するユニークな顧客IDを抽出
    # 重複を排除し、最近のものを優先するためにIDでソート
    subquery = db.session.query(
        AnalysisRecord.patient_id,
        db.func.max(AnalysisRecord.id).label('max_id')
    ).filter(
        AnalysisRecord.user_id == current_user.id,
        AnalysisRecord.patient_id.ilike(f'%{query}%')
    ).group_by(AnalysisRecord.patient_id).subquery()

    results = db.session.query(subquery.c.patient_id)\
        .order_by(subquery.c.max_id.desc())\
        .limit(10).all()
    
    return jsonify([r.patient_id for r in results])

@app.route('/api/patient_stats')
@login_required
def api_patient_stats():
    patient_id = request.args.get('patient_id')
    if not patient_id:
        return jsonify({'success': False, 'error': '顧客IDが必要です。'}), 400
    
    # 現在のログインユーザー（先生）が保存した、この患者のレコードを古い順に取得
    records = AnalysisRecord.query.filter_by(
        user_id=current_user.id, 
        patient_id=patient_id
    ).order_by(AnalysisRecord.created_at.asc()).all()
    
    data = []
    for r in records:
        jst_time = r.created_at + timedelta(hours=9)
        data.append({
            'date': jst_time.strftime('%Y-%m-%d %H:%M'),
            'view': r.view_type,
            'shoulder_angle': r.shoulder_angle,
            'pelvis_angle': r.pelvis_angle,
            'head_angle': r.head_angle,
            'ear_shift': r.ear_shift_pct,
            'fhp': r.fhp_pct,
            'rs': r.rs_pct,
            'trunk': r.trunk_pct
        })
    
    # 重複する日付やデータが多い場合に備え、ビューごとに分けたデータも検討可能ですが
    # フロントエンド側でフィルタリングする方が柔軟。
    return jsonify({'success': True, 'data': data})

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/download/<path:filename>')
@login_required
def download_image(filename):
    # 会員チェック
    if not current_user.is_active_member:
        return jsonify({'error': 'サブスクリプションの契約が必要です。'}), 403
    # static/uploads ディレクトリからファイルを強制ダウンロードとして返す
    directory = os.path.join(app.root_path, 'static', 'uploads')
    # Content-Disposition ヘッダーを付けて、ブラウザに保存を促す
    return send_from_directory(
        directory, 
        filename, 
        as_attachment=True,
        download_name=f"姿勢解析レポート_{uuid.uuid4().hex[:8]}.jpg"
    )

# ─── 管理者機能 ──────────────────────────────────────────────────────────────
@app.route('/admin')
@login_required
def admin():
    if not current_user.is_admin:
        return redirect(url_for('index'))
    
    # 統計データの取得
    now = datetime.utcnow()
    today_start = datetime(now.year, now.month, now.day)
    month_start = datetime(now.year, now.month, 1)
    
    stats = {
        'today': AnalysisLog.query.filter(AnalysisLog.created_at >= today_start).count(),
        'month': AnalysisLog.query.filter(AnalysisLog.created_at >= month_start).count(),
        'total': AnalysisLog.query.count(),
        'active_users': User.query.filter_by(is_active_member=True).count(),
        'now': now
    }
    
    users = User.query.all()
    return render_template('admin.html', users=users, stats=stats)

@app.route('/admin/toggle/<int:user_id>', methods=['POST'])
@login_required
def admin_toggle_user(user_id):
    if not current_user.is_admin:
        return jsonify({'success': False}), 403
    user = User.query.get_or_404(user_id)
    user.is_active_member = not user.is_active_member
    db.session.commit()
    return jsonify({'success': True, 'new_status': user.is_active_member})

@app.route('/admin/unlock/<int:user_id>', methods=['POST'])
@login_required
def admin_unlock_user(user_id):
    if not current_user.is_admin:
        return jsonify({'success': False}), 403
    user = User.query.get_or_404(user_id)
    user.failed_login_attempts = 0
    user.locked_until = None
    db.session.commit()
    return jsonify({'success': True})
@login_required
def admin_register_user():
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': '権限がありません。'}), 403
    
    data = request.json
    email = data.get('email', '').strip()
    password = data.get('password', '').strip()
    
    if not email or not password:
        return jsonify({'success': False, 'error': 'メールアドレスとパスワードを入力してください。'}), 400
    
    if User.query.filter_by(email=email).first():
        return jsonify({'success': False, 'error': 'このメールアドレスは既に登録されています。'}), 400
    
    new_user = User(
        email=email,
        password=generate_password_hash(password),
        is_active_member=True,
        is_admin=False
    )
    
    db.session.add(new_user)
    db.session.commit()
    
    return jsonify({'success': True})

# ─── パスワードリセット ──────────────────────────────────────────────────────
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()
        if user:
            # トークン生成
            token = str(uuid.uuid4())
            user.reset_token = token
            user.reset_token_expiration = datetime.utcnow() + timedelta(hours=1)
            db.session.commit()
            
            # 本来はメールを送信するが、今回はデバッグ用に出力して画面に表示
            reset_url = url_for('reset_password', token=token, _external=True)
            print(f"Password reset link for {email}: {reset_url}")
            flash(f'パスワード再設定リンクを発行しました（デバッグ用）: {reset_url}', 'info')
        else:
            flash('そのメールアドレスは登録されていません。', 'danger')
        return redirect(url_for('forgot_password'))
    return render_template('forgot_password.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    user = User.query.filter_by(reset_token=token).first()
    if not user or user.reset_token_expiration < datetime.utcnow():
        flash('無効なトークン、または期限切れです。', 'danger')
        return redirect(url_for('forgot_password'))
    
    if request.method == 'POST':
        password = request.form.get('password')
        user.password = generate_password_hash(password)
        user.reset_token = None
        user.reset_token_expiration = None
        db.session.commit()
        flash('パスワードを更新しました。新しいパスワードでログインしてください。', 'success')
        return redirect(url_for('login'))
    
    return render_template('reset_password.html', token=token)

@app.route('/admin/backup')
def admin_backup():
    # トークンによる認証またはログイン済み管理者のみ
    token = request.args.get('token')
    
    if token != BACKUP_TOKEN:
        if not current_user.is_authenticated or not current_user.is_admin:
            return jsonify({'success': False, 'error': '権限がありません。'}), 403
    
    users = User.query.all()
    
    # Obsidian用のMarkdownテーブル作成
    md_content = "# 会員リストバックアップ\n\n"
    md_content += "| ID | メールアドレス | 契約ステータス | 管理者権限 |\n"
    md_content += "|---|---|---|---|\n"
    
    for user in users:
        status = "契約中" if user.is_active_member else "未契約"
        admin = "はい" if user.is_admin else "いいえ"
        md_content += f"| {user.id} | {user.email} | {status} | {admin} |\n"
    
    # レスポンスとしてファイルを返す
    from flask import Response
    return Response(
        md_content,
        mimetype="text/markdown",
        headers={"Content-disposition": "attachment; filename=members_backup.md"}
    )

@app.route('/admin/export_csv')
@login_required
def admin_export_csv():
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': '権限がありません。'}), 403
    
    import csv
    import io
    from flask import make_response

    # 全ログ取得
    logs = AnalysisLog.query.order_by(AnalysisLog.created_at.desc()).all()
    
    # メモリ上にCSV出力
    output = io.StringIO()
    # Excelで文字化けしないようBOMを追加
    output.write('\ufeff')
    writer = csv.writer(output)
    
    # ヘッダー
    writer.writerow([
        'ID', '実行日時', '顧客ID/名前', 'ユーザーID/メール', '解析タイプ', 
        '肩角度', '骨盤角度', '頭部角度', '頭部ズレ%', '肩ズレ%', '骨盤ズレ%',
        'FHP%', 'ラウンド肩%', '側面骨盤角度', '体幹ズレ%'
    ])
    
    # データ
    for log in logs:
        user_info = f"UID:{log.user_id}"
        if log.user_id:
            user = User.query.get(log.user_id)
            if user:
                user_info = user.email
        
        # 紐付く数値データを取得（最新の1件）
        record = AnalysisRecord.query.filter_by(user_id=log.user_id).order_by(AnalysisRecord.created_at.desc()).first()
        # ※本来はLogとRecordを1対1で紐付けるべきですが、現在の簡易実装では同時刻のものを当てるか、
        # 実行時の ID を保持するように拡張するのが理想的です。
        # 今回は、最新のデータエクスポートとして、Recordテーブルをベースにする形に書き換えます。
        
    # より正確に数値データを出すため、AnalysisRecordベースで出力するように変更
    records = AnalysisRecord.query.order_by(AnalysisRecord.created_at.desc()).all()
    for rec in records:
        user_info = f"UID:{rec.user_id}"
        if rec.user_id:
            user = User.query.get(rec.user_id)
            if user:
                user_info = user.email
        
        jst_time = rec.created_at + timedelta(hours=9)
        writer.writerow([
            rec.id,
            jst_time.strftime('%Y-%m-%d %H:%M:%S'),
            rec.patient_id,
            user_info,
            rec.view_type,
            rec.shoulder_angle,
            rec.pelvis_angle,
            rec.head_angle,
            rec.ear_shift_pct,
            rec.shoulder_shift_pct,
            rec.pelvis_shift_pct,
            rec.fhp_pct,
            rec.rs_pct,
            rec.side_pelvis_angle,
            rec.trunk_pct
        ])
    
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = f"attachment; filename=analysis_logs_{datetime.now().strftime('%Y%m%d')}.csv"
    response.headers["Content-type"] = "text/csv; charset=utf-8-sig"
    return response

def generate_advice(record):
    """
    解析レコードの数値に基づいて、日本語のアドバイスを生成する
    """
    advices = []
    
    # 正面解析のアドバイス
    if record.view_type == 'front':
        # 肩の傾き
        if record.shoulder_angle and abs(record.shoulder_angle) > 3.0:
            advices.append("肩の左右バランスに傾きが見られます。片側の筋肉の緊張や、鞄の持ち方の癖などが影響している可能性があります。")
        elif record.shoulder_angle and abs(record.shoulder_angle) > 1.5:
            advices.append("肩のラインにわずかな左右差があります。ストレッチや日常の姿勢に気をつけましょう。")
        
        # 骨盤の傾き
        if record.pelvis_angle and abs(record.pelvis_angle) > 3.0:
            advices.append("骨盤の傾きが顕著です。腰痛や膝への負担に繋がるほか、足の長さの左右差を感じる原因になります。")
        elif record.pelvis_angle and abs(record.pelvis_angle) > 1.5:
            advices.append("骨盤に少し傾きがあります。座り方（足を組む等）の習慣を見直すのが効果的です。")
        
        # 頭の傾き
        if record.head_angle and abs(record.head_angle) > 3.0:
            advices.append("頭部の傾きが大きく、首すじの筋肉（胸鎖乳突筋など）に負担がかかりやすい状態です。")
            
    # 側面解析のアドバイス
    elif record.view_type == 'side':
        # FHP (首の突き出し)
        if record.fhp_pct and record.fhp_pct > 10.0:
            advices.append("頭部が前方へ強く突き出しています（スマホ首）。首や肩こり、頭痛の主な原因となります。アゴを引く意識が大切です。")
        elif record.fhp_pct and record.fhp_pct > 5.0:
            advices.append("頭がやや前方に出る傾向があります。デスクワーク時のディスプレイの高さなどを調整してみましょう。")
            
        # 巻肩 (RS)
        if record.rs_pct and record.rs_pct > 15.0:
            advices.append("強い巻き肩の状態です。胸の筋肉が縮み、呼吸が浅くなったり、背中が張りやすくなったりします。")
        elif record.rs_pct and record.rs_pct > 8.0:
            advices.append("少し巻き肩気味です。肩甲骨を寄せるように胸を開くストレッチが有効です。")
            
        # 体幹のズレ
        if record.trunk_pct and abs(record.trunk_pct) > 5.0:
            advices.append("重心が前後にズレており、腰や足裏への負担が不均等になっています。")

    # 共通の締めくくり
    if not advices:
        if record.view_type == 'compare':
            advices.append("比較解析を行いました。以前の状態（左）と今回の状態（右）の変化を確認してください。")
        else:
            advices.append("全体的に非常に良好な姿勢です。この状態を維持していきましょう！")
    else:
        advices.append("まずは意識して姿勢を正すことから始め、定期的なメンテナンスをお勧めします。")
        
    return "\n".join(advices)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    view_type = request.form.get('view_type', 'auto') # 'front', 'side', or 'auto'
    patient_id = request.form.get('patient_id', '')

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # ファイル名をユニークにする
    ext = os.path.splitext(file.filename)[1]
    filename = f"{uuid.uuid4()}{ext}"
    input_path = os.path.join(UPLOAD_FOLDER, f"input_{filename}")
    output_path = os.path.join(UPLOAD_FOLDER, f"report_{filename}")
    
    file.save(input_path)

    try:
        # 新しいクラスベースの解析実行
        res = get_analyzer().analyze(input_path, output_path, view_type=view_type)

        if res and res.get('success'):
            # Cloudinaryへのアップロード
            cloud_url = upload_to_cloudinary(output_path)
            input_cloud_url = upload_to_cloudinary(input_path)

            # ログ記録
            log = AnalysisLog(user_id=current_user.id if current_user.is_authenticated else None, view_type=res.get('view', view_type))
            db.session.add(log)
            
            # 数値データの保存
            try:
                data = res.get('data', {})
                record = AnalysisRecord(
                    user_id=current_user.id if current_user.is_authenticated else None,
                    patient_id=patient_id,
                    view_type=res.get('view', view_type),
                    shoulder_angle=data.get('shoulder_angle'),
                    pelvis_angle=data.get('pelvis_angle'),
                    head_angle=data.get('head_angle'),
                    ear_shift_pct=data.get('ear_shift_pct'),
                    shoulder_shift_pct=data.get('shoulder_shift_pct'),
                    pelvis_shift_pct=data.get('pelvis_shift_pct'),
                    fhp_pct=data.get('fhp_pct'),
                    rs_pct=data.get('rs_pct'),
                    side_pelvis_angle=data.get('pelvis_angle') if res.get('view') == 'side' else None,
                    trunk_pct=data.get('trunk_pct'),
                    image_filename=cloud_url if cloud_url else f"report_{filename}",
                    input_filename=input_cloud_url if input_cloud_url else f"input_{filename}"
                )
                
                # AIアドバイス生成
                record.advice = generate_advice(record)
                
                db.session.add(record)
            except Exception as e:
                print(f"Failed to save numerical data: {e}")
            
            db.session.commit()
            
            return jsonify({
                'success': True,
                'report_url': cloud_url if cloud_url else url_for('static', filename=f'uploads/report_{filename}'),
                'advice': record.advice
            })
        else:
            return jsonify({'success': False, 'error': '人物が検出されませんでした。'}), 200
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

# ─── 顧客管理機能 ────────────────────────────────────────────────────────────

@app.route('/api/line/send_report', methods=['POST'])
@login_required
def send_line_report():
    record_id = request.form.get('record_id')
    line_user_id = request.form.get('line_user_id')
    
    if not record_id or not line_user_id:
        return jsonify({'success': False, 'error': '必要な情報が不足しています。'}), 400
    
    record = AnalysisRecord.query.get(record_id)
    if not record:
        return jsonify({'success': False, 'error': '指定された記録が見つかりません。'}), 404

    # LINEへのメッセージ送信
    # ユーザー個別のトークンを優先、なければ環境変数のトークンを使用
    token = current_user.line_access_token or app.config.get('LINE_CHANNEL_ACCESS_TOKEN')
    
    if not token:
        # トークンが全く設定されていない場合はモック動作（開発用）
        app.logger.warning(f"LINE token is not set for user {current_user.id}. Skipping actual send.")
        record.line_user_id = line_user_id
        db.session.commit()
        return jsonify({'success': True, 'warning': 'LINE APIトークンが設定されていないため、システム上の記録のみ更新しました。マイページから設定を行うと実際に送信されます。'})

    try:
        from linebot import LineBotApi
        from linebot.models import TextSendMessage
        
        line_bot_api = LineBotApi(token)
        # レポートURLの構築（本番環境のドメインに合わせて調整が必要な場合あり）
        report_url = f"{request.host_url}patient/{record.patient_id}"
        message = f"【整体院 導】姿勢解析レポートが届きました。\n以下のリンクからご確認いただけます：\n{report_url}\n\n※このメッセージには返信できません。"
        
        line_bot_api.push_message(line_user_id, TextSendMessage(text=message))
        
        # IDを保存
        record.line_user_id = line_user_id
        db.session.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        app.logger.error(f"LINE send error: {e}")
        return jsonify({'success': False, 'error': f'送信に失敗しました: {str(e)}'}), 500

@app.route('/api/line/search_users', methods=['GET'])
@login_required
def search_line_users():
    query = request.args.get('q', '')
    # ログイン中のユーザー（院）が収集したマッピングのみ検索
    mappings = LineUserMapping.query.filter(
        LineUserMapping.owner_id == current_user.id,
        LineUserMapping.display_name.ilike(f'%{query}%')
    ).order_by(LineUserMapping.updated_at.desc()).limit(10).all()
    
    return jsonify([{
        'line_user_id': m.line_user_id,
        'display_name': m.display_name,
        'updated_at': m.updated_at.strftime('%Y-%m-%d %H:%M')
    } for m in mappings])


@app.route('/callback/<int:user_id>', methods=['POST'])
def user_callback(user_id):
    from linebot import LineBotApi
    user = User.query.get(user_id)
    if not user or not user.line_access_token:
        return 'User configuration missing', 400

    body = request.get_data(as_text=True)
    signature = request.headers.get('X-Line-Signature')
    
    # 署名検証略（後で追加可能）
    try:
        data = json.loads(body)
        line_bot_api = LineBotApi(user.line_access_token)
        
        for event in data.get('events', []):
            line_user_id = event['source']['userId']
            
            # ユーザー情報の取得
            try:
                profile = line_bot_api.get_profile(line_user_id)
                display_name = profile.display_name
                
                # マッピングの保存・更新
                mapping = LineUserMapping.query.filter_by(
                    line_user_id=line_user_id, 
                    owner_id=user_id
                ).first()
                
                if not mapping:
                    mapping = LineUserMapping(
                        line_user_id=line_user_id,
                        display_name=display_name,
                        owner_id=user_id
                    )
                    db.session.add(mapping)
                else:
                    mapping.display_name = display_name
                
                db.session.commit()
            except Exception as e:
                app.logger.error(f"Failed to get LINE profile: {e}")
                
    except Exception as e:
        app.logger.error(f"Callback error: {e}")
        
    return 'OK'

# ─── LINE ユーザーマッピングAPI ──────────────────────────────────────────────────

@app.route('/api/line/mapping/list', methods=['GET'])
@login_required
def get_line_mappings():
    mappings = LineUserMapping.query.filter_by(owner_id=current_user.id).order_by(LineUserMapping.display_name).all()
    result = []
    for m in mappings:
        result.append({
            'id': m.id,
            'display_name': m.display_name,
            'line_user_id': m.line_user_id,
            'updated_at': m.updated_at.strftime('%Y-%m-%d %H:%M')
        })
    return jsonify(result)

@app.route('/api/line/mapping/add', methods=['POST'])
@login_required
def add_line_mapping():
    display_name = request.form.get('display_name')
    line_user_id = request.form.get('line_user_id')
    
    if not display_name or not line_user_id:
        return jsonify({'success': False, 'error': '名前とLINE IDは必須です。'}), 400
    
    # 重複チェック（同一オーナー内）
    existing = LineUserMapping.query.filter_by(line_user_id=line_user_id, owner_id=current_user.id).first()
    if existing:
        existing.display_name = display_name
        existing.updated_at = db.func.current_timestamp()
    else:
        mapping = LineUserMapping(
            display_name=display_name,
            line_user_id=line_user_id,
            owner_id=current_user.id
        )
        db.session.add(mapping)
    
    db.session.commit()
    return jsonify({'success': True})

@app.route('/api/line/mapping/delete/<int:mapping_id>', methods=['POST'])
@login_required
def delete_line_mapping(mapping_id):
    mapping = LineUserMapping.query.get_or_404(mapping_id)
    if mapping.owner_id != current_user.id:
        return jsonify({'success': False, 'error': '権限がありません。'}), 403
    
    db.session.delete(mapping)
    db.session.commit()
    return jsonify({'success': True})

@app.route('/patients')
@login_required
def patients():
    # 重複を除去したpatient_idのリストを取得し、それぞれの最新解析日も取得
    # patient_idが空のものは除外
    records = db.session.query(
        AnalysisRecord.patient_id, 
        db.func.max(AnalysisRecord.created_at).label('last_visit')
    ).filter(AnalysisRecord.patient_id != '').group_by(AnalysisRecord.patient_id).all()
    
    # ソート引数
    sort_by = request.args.get('sort', 'name') # 'name', 'visit'
    order = request.args.get('order', 'asc')
    
    # recordsは [(id, date), ...] の形式
    patient_list = []
    for r in records:
        patient_list.append({'id': r[0], 'last_visit': r[1]})
        
    if sort_by == 'visit':
        patient_list.sort(key=lambda x: x['last_visit'], reverse=(order == 'desc'))
    else: # nameソート
        patient_list.sort(key=lambda x: x['id'], reverse=(order == 'desc'))
        
    return render_template('patients.html', patients=patient_list, sort_by=sort_by, order=order)

@app.route('/patient/<patient_id>')
@login_required
def patient_detail(patient_id):
    # 特定の顧客の履歴を全件取得（新しい順）
    records = AnalysisRecord.query.filter_by(patient_id=patient_id).order_by(AnalysisRecord.created_at.desc()).all()
    if not records:
        flash("顧客データが見つかりませんでした。")
        return redirect(url_for('patients'))
    
    return render_template('patient_detail.html', patient_id=patient_id, records=records)

@app.route('/record/memo/<int:record_id>', methods=['POST'])
@login_required
def update_memo(record_id):
    record = AnalysisRecord.query.get_or_404(record_id)
    # 簡易的な所有権チェック（通常は user_id を使いますが、ここでは patient_id を重視）
    memo_text = request.form.get('memo', '')
    record.memo = memo_text
    db.session.commit()
    return jsonify({'success': True})

@app.route('/record/delete/<int:record_id>', methods=['POST'])
@login_required
def delete_record(record_id):
    record = AnalysisRecord.query.get_or_404(record_id)
    patient_id = record.patient_id
    db.session.delete(record)
    db.session.commit()
    
    # もしその顧客のデータが他に残っていなければ、一覧へ戻す
    remaining = AnalysisRecord.query.filter_by(patient_id=patient_id).count()
    if remaining == 0:
        return jsonify({'success': True, 'redirect': url_for('patients')})
    return jsonify({'success': True})

# ─── 比較解析機能 ────────────────────────────────────────────────────────────

@app.route('/compare', methods=['POST'])
@login_required
def compare():
    if 'image_before' not in request.files or 'image_after' not in request.files:
        return jsonify({'error': 'Before/After both images are required'}), 400
    
    file_b = request.files['image_before']
    file_a = request.files['image_after']
    view_type = request.form.get('view_type', 'auto')
    patient_id = request.form.get('patient_id', '')

    # ユニークファイル名生成
    uid = uuid.uuid4().hex[:8]
    ext_b = os.path.splitext(file_b.filename)[1] or ".jpg"
    ext_a = os.path.splitext(file_a.filename)[1] or ".jpg"
    
    path_b = os.path.join(UPLOAD_FOLDER, f"comp_b_{uid}{ext_b}")
    path_a = os.path.join(UPLOAD_FOLDER, f"comp_a_{uid}{ext_a}")
    output_path = os.path.join(UPLOAD_FOLDER, f"report_comp_{uid}.jpg")
    
    file_b.save(path_b); file_a.save(path_a)

    try:
        res = get_analyzer().analyze_comparison(path_b, path_a, output_path, view_type=view_type)
        if res and res.get('success'):
            data = res.get('data', {})
            view = data.get('view', view_type)
            
            # ログ記録
            log = AnalysisLog(user_id=current_user.id if current_user.is_authenticated else None, view_type='compare')
            db.session.add(log)
            
            # 数値データの保存（BeforeとAfter両方保存する例）
            def save_comp_data(items, prefix_type):
                # items is a list of {"n": name, "v": value, "s": score}
                # mapping to columns
                d = {it['n']: it['v'] for it in items}
                # Cloudinaryへのアップロード
                c_url = upload_to_cloudinary(output_path)
                # Before/Afterに応じて入力を選択
                source_path = path_b if prefix_type == 'before' else path_a
                i_url = upload_to_cloudinary(source_path)

                record = AnalysisRecord(
                    user_id=current_user.id if current_user.is_authenticated else None,
                    patient_id=patient_id,
                    view_type=f"{view}_{prefix_type}",
                    shoulder_angle=d.get('肩傾き') or d.get('ラウンド肩'),
                    pelvis_angle=d.get('骨盤傾き') or d.get('骨盤前後傾'),
                    head_angle=d.get('頭部傾き'),
                    ear_shift_pct=d.get('頭部ズレ'),
                    shoulder_shift_pct=d.get('肩部ズレ'),
                    pelvis_shift_pct=d.get('骨盤ズレ'),
                    fhp_pct=d.get('FHP'),
                    rs_pct=d.get('ラウンド肩'),
                    # side_pelvis_angle は pelvis_angle と共通化
                    trunk_pct=d.get('体幹領域') or d.get('体幹ライン'),
                    image_filename=c_url if c_url else os.path.basename(output_path),
                    input_filename=i_url if i_url else (os.path.basename(path_b) if prefix_type == 'before' else os.path.basename(path_a))
                )
                
                # アドバイス
                record.advice = generate_advice(record)
                
                db.session.add(record)

            try:
                save_comp_data(data['before'], 'before')
                save_comp_data(data['after'], 'after')
            except Exception as e:
                print(f"Failed to save comparison numerical data: {e}")

            db.session.commit()

            return jsonify({
                'success': True,
                'report_url': url_for('static', filename=f'uploads/report_comp_{uid}.jpg'),
                'advice': "比較解析を行いました。以前の状態（左）と今回の状態（右）の変化を確認してください。"
            })
        else:
            return jsonify({'success': False, 'error': '人物の検出に失敗しました。'}), 200
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # ポート番号はクラウド環境（Render等）の指定に従い、なければ 5001 を使用
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)
