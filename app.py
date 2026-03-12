import os
import sys
import uuid
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for, redirect, flash
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import sqlalchemy
from sqlalchemy import text
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

# 姿勢解析エンジンのインポート
from pose_analyzer import PoseAnalyzer

app = Flask(__name__)
# 環境変数から取得
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'seitai-michibiki-secret-key-12345')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# バックアップ用トークンも環境変数から取得
BACKUP_TOKEN = os.environ.get('BACKUP_TOKEN', 'seitai-backup-2026-safe')

CORS(app)
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', code=404, title="Page Not Found"), 404

@app.errorhandler(500)
def server_error(e):
    if request.is_json:
        return jsonify({'success': False, 'error': 'Internal Server Error'}), 500
    return render_template('error.html', code=500, title="Server Error"), 500

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login' # ログインしていない場合に飛ばす先

# ─── データベース初期化・自動マイグレーション ────────────────────────────────
def init_and_migrate():
    with app.app_context():
        # Userモデルなどが定義された後にdb.create_all()を呼ぶ必要がある
        db.create_all()
        # SQLite用の簡易的なカラム追加チェック
        try:
            inspector = sqlalchemy.inspect(db.engine)
            columns = [c['name'] for c in inspector.get_columns('user')]
            
            # 不要なカラム追加を避けるためのチェック
            if 'reset_token' not in columns:
                with db.engine.connect() as conn:
                    conn.execute(text('ALTER TABLE user ADD COLUMN reset_token VARCHAR(100)'))
                    conn.execute(text('ALTER TABLE user ADD COLUMN reset_token_expiration DATETIME'))
                    conn.commit()
                print("Database migrated: Added password reset columns.")
            else:
                print("Database already up to date (reset token).")
            
            # ログイン試行制限用のカラム追加
            if 'failed_login_attempts' not in columns:
                with db.engine.connect() as conn:
                    conn.execute(text('ALTER TABLE user ADD COLUMN failed_login_attempts INTEGER DEFAULT 0'))
                    conn.execute(text('ALTER TABLE user ADD COLUMN locked_until DATETIME'))
                    conn.commit()
                print("Database migrated: Added login attempt limiting columns.")
        except Exception as e:
            # ログに出力
            app.logger.error(f"Migration error: {e}")
            print(f"Migration notice: {e}")

# ─── データベースモデル ──────────────────────────────────────────────────────
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    is_active_member = db.Column(db.Boolean, default=False) # サブスク有効フラグ
    is_admin = db.Column(db.Boolean, default=False)
    # パスワードリセット用
    reset_token = db.Column(db.String(100), unique=True, nullable=True)
    reset_token_expiration = db.Column(db.DateTime, nullable=True)
    # ログイン試行制限用
    failed_login_attempts = db.Column(db.Integer, default=0)
    locked_until = db.Column(db.DateTime, nullable=True)

class AnalysisLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    view_type = db.Column(db.String(20), nullable=False) # 'front', 'side', 'compare'
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
                login_user(user)
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
    writer.writerow(['ID', '実行日時', 'ユーザーID/メール', '解析タイプ'])
    
    # データ
    for log in logs:
        user_info = f"UID:{log.user_id}"
        if log.user_id:
            user = User.query.get(log.user_id)
            if user:
                user_info = user.email
        
        # 日本時間に調整（簡易的）
        jst_time = log.created_at + timedelta(hours=9)
        writer.writerow([
            log.id,
            jst_time.strftime('%Y-%m-%d %H:%M:%S'),
            user_info,
            log.view_type
        ])
    
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = f"attachment; filename=analysis_logs_{datetime.now().strftime('%Y%m%d')}.csv"
    response.headers["Content-type"] = "text/csv; charset=utf-8-sig"
    return response

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    view_type = request.form.get('view_type', 'auto') # 'front', 'side', or 'auto'

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
        success = get_analyzer().analyze(input_path, output_path, view_type=view_type)

        if success:
            # ログ記録
            log = AnalysisLog(user_id=current_user.id if current_user.is_authenticated else None, view_type=view_type)
            db.session.add(log)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'report_url': url_for('static', filename=f'uploads/report_{filename}')
            })
        else:
            return jsonify({'success': False, 'error': '人物が検出されませんでした。'}), 200
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/compare', methods=['POST'])
def compare():
    if 'image_before' not in request.files or 'image_after' not in request.files:
        return jsonify({'error': 'Before/After both images are required'}), 400
    
    file_b = request.files['image_before']
    file_a = request.files['image_after']
    view_type = request.form.get('view_type', 'auto')

    # ユニークファイル名生成
    uid = uuid.uuid4().hex[:8]
    ext_b = os.path.splitext(file_b.filename)[1] or ".jpg"
    ext_a = os.path.splitext(file_a.filename)[1] or ".jpg"
    
    path_b = os.path.join(UPLOAD_FOLDER, f"comp_b_{uid}{ext_b}")
    path_a = os.path.join(UPLOAD_FOLDER, f"comp_a_{uid}{ext_a}")
    output_path = os.path.join(UPLOAD_FOLDER, f"report_comp_{uid}.jpg")
    
    file_b.save(path_b); file_a.save(path_a)

    try:
        success = get_analyzer().analyze_comparison(path_b, path_a, output_path, view_type=view_type)
        if success:
            # ログ記録
            log = AnalysisLog(user_id=current_user.id if current_user.is_authenticated else None, view_type='compare')
            db.session.add(log)
            db.session.commit()

            return jsonify({
                'success': True,
                'report_url': url_for('static', filename=f'uploads/report_comp_{uid}.jpg')
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
