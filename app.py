import os
import sys
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for, redirect
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

# 姿勢解析エンジンのインポート
from pose_analyzer import PoseAnalyzer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'seitai-michibiki-secret-key-12345' # 本番環境では環境変数から取得推奨
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app)
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login' # ログインしていない場合に飛ばす先

# ─── データベースモデル ──────────────────────────────────────────────────────
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    is_active_member = db.Column(db.Boolean, default=False) # サブスク有効フラグ
    is_admin = db.Column(db.Boolean, default=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

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
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'メールアドレスまたはパスワードが正しくありません。'}), 401
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

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
    users = User.query.all()
    return render_template('admin.html', users=users)

@app.route('/admin/toggle/<int:user_id>', methods=['POST'])
@login_required
def toggle_user(user_id):
    if not current_user.is_admin:
        return jsonify({'success': False}), 403
    user = User.query.get_or_404(user_id)
    user.is_active_member = not user.is_active_member
    db.session.commit()
    return jsonify({'success': True, 'new_status': user.is_active_member})

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
