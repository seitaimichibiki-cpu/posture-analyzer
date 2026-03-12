import os
import sys
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from flask_cors import CORS

# 姿勢解析エンジンのインポートと初期化
from pose_analyzer import PoseAnalyzer

app = Flask(__name__)
CORS(app)

# 設定
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

@app.route('/')
def index():
    return render_template('index.html')

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
        import traceback
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # ポート番号はクラウド環境（Render等）の指定に従い、なければ 5001 を使用
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)
