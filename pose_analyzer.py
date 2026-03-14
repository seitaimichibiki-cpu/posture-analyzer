import os
import cv2
import math
import numpy as np
import mediapipe as mp
import glob
import uuid
import json
from PIL import Image, ImageDraw, ImageFont

# ─── スタイル・定数 ─────────────────────────────────────────────────────────
PANEL_BG = (24, 29, 48)
PANEL_FG = (230, 235, 255)
LINE_COL = (60, 80, 150)
WHITE = (255, 255, 255)
GRAY = (180, 185, 200)
YELLOW = (255, 215, 0)
GREEN_IDEAL = (100, 255, 120)
MIDLINE_COL_BGR = (255, 230, 100)
SCORE_RGB = {"◎": (100, 255, 120), "○": (100, 220, 255), "△": (255, 220, 100), "×": (255, 100, 100)}
SCORE_BGR = {k: (v[2], v[1], v[0]) for k, v in SCORE_RGB.items()}

# 解析しきい値
THRESHOLDS = {
    "Head": [2.0, 4.5, 7.0],
    "Shoulder": [1.5, 3.5, 6.0],
    "Pelvis": [1.0, 3.0, 5.5]
}
TRUNK_SHIFT_TH = [3.0, 6.5, 12.0]
SIDE_THRESHOLDS = {
    "FHP": [6.0, 12.0, 18.0],
    "Round": [5.0, 10.0, 16.0],
    "Trunk": [4.0, 8.0, 14.0],
    "Pelvis": [3.0, 7.0, 12.0]
}

# ─── 姿勢解析エンジン ───────────────────────────────────────
class PoseAnalyzer:
    def __init__(self, model_path):
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE
        )
        self.landmarker = PoseLandmarker.create_from_options(self.options)

    def analyze(self, image_path, output_path, view_type='auto'):
        """
        view_type: 'front', 'side', or 'auto'
        """
        img = cv2.imread(image_path)
        if img is None: return {'success': False, 'error': 'Image load failed'}
        h, w = img.shape[:2]
        
        mp_image = mp.Image.create_from_file(image_path)
        res = self.landmarker.detect(mp_image)
        
        if not res.pose_landmarks:
            return {'success': False, 'error': 'Person not detected'}
            
        lm = res.pose_landmarks[0]
        
        # 向き判定
        actual_view = view_type
        if view_type == 'auto':
            # 肩の幅と腰の幅の比率などで簡易判定
            l_sh, r_sh = lm[11], lm[12]
            sh_dist = abs(l_sh.x - r_sh.x)
            actual_view = 'side' if sh_dist < 0.12 else 'front'

        if actual_view == 'front':
            return self._analyze_front(img, lm, w, h, output_path)
        else:
            return self._analyze_side(img, lm, w, h, output_path)

    def _analyze_front(self, img, lm, w, h, output_path):
        # 計測
        sha = calc_angle(lm[11], lm[12], w, h)
        pla = calc_angle(lm[23], lm[24], w, h)
        hda = calc_angle(lm[7], lm[8], w, h)
        
        # 正中線偏差 (左右の足首の中点を基準)
        base_x = (lm[27].x + lm[28].x) / 2
        nose_x = lm[0].x
        trunk_shift = ((nose_x - base_x) * w / (w * 0.15)) * 10 
        ts_ratio = abs(trunk_shift)

        items = [
            {"name": "頭部の傾き", "score": get_score("Head", abs(hda)), "measured": hda, "normal": 0.0, "diff": abs(hda), "direction": direction(hda)},
            {"name": "肩のライン", "score": get_score("Shoulder", abs(sha)), "measured": sha, "normal": 0.0, "diff": abs(sha), "direction": direction(sha)},
            {"name": "骨盤のライン", "score": get_score("Pelvis", abs(pla)), "measured": pla, "normal": 0.0, "diff": abs(pla), "direction": direction(pla)},
            {"name": "体幹の垂直性", "score": get_trunk_score(ts_ratio), "measured": ts_ratio, "normal": 0.0, "diff": ts_ratio, "direction": "右ズレ" if trunk_shift > 0 else "左ズレ"}
        ]
        
        risks = calc_body_risks(items[0]["score"], items[1]["score"], items[2]["score"], items[3]["score"], sha, pla)
        
        # 描画 (ズーム & クロップ)
        annotated = self._draw_front_annotated(img, lm, w, h, items)
        
        # パネル作成
        panel = build_panel(items, risks, 480, annotated.shape[0])
        
        # 合体
        self._save_final_report(annotated, panel, output_path, "正面観察")
        
        return {
            'success': True, 
            'view': 'front',
            'data': {
                'shoulder_angle': sha, 'pelvis_angle': pla, 'head_angle': hda, 'trunk_pct': trunk_shift
            }
        }

    def _analyze_side(self, img, lm, w, h, output_path):
        # 基準点：足首
        base_x = (lm[27].x + lm[28].x) / 2
        
        # FHP (耳-肩)
        ear = lm[7] if lm[7].visibility > lm[8].visibility else lm[8]
        shoulder = lm[11] if lm[11].visibility > lm[12].visibility else lm[12]
        fhp_val = (ear.x - shoulder.x) * w / (h * 0.1) * 10
        fhp_sc = _get_side_score("FHP", abs(fhp_val))
        
        # Round Shoulder (肩-体幹重心)
        hip = lm[23] if lm[23].visibility > lm[24].visibility else lm[24]
        trunk_x = (shoulder.x + hip.x) / 2
        rs_val = (shoulder.x - trunk_x) * w / (h * 0.1) * 10
        rs_sc = _get_side_score("Round", abs(rs_val))
        
        # Trunk Shift (重心ライン)
        center_x = (shoulder.x + hip.x) / 2
        ts_val = (center_x - base_x) * w / (h * 0.2) * 10
        ts_sc = _get_side_score("Trunk", abs(ts_val))
        
        # Pelvic Tilt (簡易)
        knee = lm[25] if lm[25].visibility > lm[26].visibility else lm[26]
        pel_angle = calc_angle(hip, knee, w, h) - 90 # 垂直からのズレ
        pel_sc = _get_side_score("Pelvis", abs(pel_angle))

        items = [
            {"name": "首の位置(FHP)", "score": fhp_sc, "measured": f"約{abs(fhp_val):.1f}%", "ideal": "垂直線上", "diff": f"{abs(fhp_val):.1f}%"},
            {"name": "肩の巻き込み", "score": rs_sc, "measured": f"約{abs(rs_val):.1f}%", "ideal": "中心線上", "diff": f"{abs(rs_val):.1f}%"},
            {"name": "重心のズレ", "score": ts_sc, "measured": f"約{abs(ts_val):.1f}%", "ideal": "垂直一直線", "diff": f"{abs(ts_val):.1f}%"},
            {"name": "骨盤の傾斜", "score": pel_sc, "measured": f"{abs(pel_angle):.1f}°", "ideal": "0.0°", "diff": f"{abs(pel_angle):.1f}°"}
        ]
        
        risks = calc_side_risks(fhp_sc, rs_sc, ts_sc, pel_sc, fhp_val, rs_val)
        
        # 描画
        annotated = self._draw_side_annotated(img, lm, w, h, items)
        panel = build_side_panel(items, risks, 480, annotated.shape[0])
        
        self._save_final_report(annotated, panel, output_path, "側面観察")
        
        return {
            'success': True,
            'view': 'side',
            'data': {
                'fhp_pct': fhp_val, 'rs_pct': rs_val, 'trunk_pct': ts_val, 'pelvis_angle': pel_angle
            }
        }

    def _draw_front_annotated(self, img, lm, w, h, items):
        # 体全体が入るようにクロップ
        ymin = min(l.y for l in lm) * h; ymax = max(l.y for l in lm) * h
        xmin = min(l.x for l in lm) * w; xmax = max(l.x for l in lm) * w
        
        # マージン
        pad = 0.15 * h
        y1, y2 = int(max(0, ymin - pad)), int(min(h, ymax + pad * 0.5))
        # 縦横比を一定にする (4:5)
        target_w = (y2 - y1) * 0.8
        cx = (xmin + xmax) / 2
        x1, x2 = int(max(0, cx - target_w/2)), int(min(w, cx + target_w/2))
        
        cropped = img[y1:y2, x1:x2].copy()
        cw, ch = cropped.shape[1], cropped.shape[0]
        scalex = cw / (x2 - x1) if x2-x1 > 0 else 1
        
        # 倍率
        scale = cw / (x2 - x1)
        
        draw_skeleton_zoom(cropped, lm, w, h, x1, y1, scale)
        # ライン描画
        draw_meas_line_zoom(cropped, lm[11], lm[12], items[1]["score"], w, h, x1, y1, scale)
        draw_meas_line_zoom(cropped, lm[23], lm[24], items[2]["score"], w, h, x1, y1, scale)
        draw_meas_line_zoom(cropped, lm[7], lm[8], items[0]["score"], w, h, x1, y1, scale)
        draw_midline_zoom(cropped, lm, w, h, x1, y1, scale)
        
        # 下部に凡例
        cv2.rectangle(cropped, (0, ch-40), (cw, ch), (20,20,30), -1)
        draw_cog_indicator(cropped, lm, 'front', w, h, x1, y1, scale)
        
        return cropped

    def _draw_side_annotated(self, img, lm, w, h, items):
        ymin = min(l.y for l in lm) * h; ymax = max(l.y for l in lm) * h
        xmin = min(l.x for l in lm) * w; xmax = max(l.x for l in lm) * w
        pad = 0.15 * h
        y1, y2 = int(max(0, ymin - pad)), int(min(h, ymax + pad * 0.5))
        target_w = (y2 - y1) * 0.8
        cx = (xmin + xmax) / 2
        x1, x2 = int(max(0, cx - target_w/2)), int(min(w, cx + target_w/2))
        
        cropped = img[y1:y2, x1:x2].copy()
        cw, ch = cropped.shape[1], cropped.shape[0]
        scale = cw / (x2 - x1)
        
        draw_skeleton_zoom(cropped, lm, w, h, x1, y1, scale)
        
        # 側面・垂直ライン (耳から足首へ)
        ank = midpoint(px_zoom(lm[27],w,h,x1,y1,scale), px_zoom(lm[28],w,h,x1,y1,scale))
        ear = midpoint(px_zoom(lm[7],w,h,x1,y1,scale), px_zoom(lm[8],w,h,x1,y1,scale))
        cv2.line(cropped, (ank[0], 20), (ank[0], ch-20), (80,220,240), 2, cv2.LINE_AA)
        cv2.circle(cropped, ank, 8, (80,220,240), -1)
        cv2.circle(cropped, ear, 8, SCORE_BGR[items[0]["score"]], -1)
        
        # 補助矢印
        if abs(ear[0] - ank[0]) > 5:
            cv2.arrowedLine(cropped, (ank[0], ear[1]), (ear[0], ear[1]), (255,255,255), 2, tipLength=0.3)

        draw_cog_indicator(cropped, lm, 'side', w, h, x1, y1, scale)
        return cropped

    def _save_final_report(self, img, panel, output_path, title_suffix):
        # 高さを揃える
        ih, iw = img.shape[:2]
        ph, pw = panel.shape[:2]
        
        max_h = max(ih, ph)
        def pad_img(im, target_h, bg_col):
            curr_h, curr_w = im.shape[:2]
            res = np.full((target_h, curr_w, 3), bg_col, dtype=np.uint8)
            res[:curr_h, :] = im
            return res
            
        img_pad = pad_img(img, max_h, (30, 35, 60))
        panel_pad = pad_img(panel, max_h, (24, 29, 48))
        
        canvas = np.hstack([img_pad, panel_pad])
        bar_h = 75; fw, fh = canvas.shape[1], canvas.shape[0] + bar_h
        full_p = Image.new("RGB", (fw, fh), (28, 34, 58))
        full_d = ImageDraw.Draw(full_p)
        draw_text_center(full_d, fw//2, 8, f"整体院 導 ｜ AI 姿勢解析レポート（{title_suffix}）", get_font(28), WHITE)
        draw_text_center(full_d, fw//2, 42, "MediaPipe Pose Estimation + エビデンスベース解析", get_font(16), GRAY)
        full_p.paste(cv2pil(canvas), (0, bar_h))
        final = pil2cv2(np.array(full_p))
        cv2.imwrite(output_path, final)
        print(f"  ✅ 保存: {os.path.basename(output_path)}")
        return True

# ─── ヘルパー関数 ───────────────────────────────────────
def calc_angle(p1, p2, w, h):
    dy, dx = (p2.y-p1.y)*h, (p2.x-p1.x)*w
    return math.degrees(math.atan2(dy, abs(dx))) if abs(dx)>1e-6 else 90.0

def get_score(part, val):
    t = THRESHOLDS[part]
    if val < t[0]: return "◎"
    if val < t[1]: return "○"
    if val < t[2]: return "△"
    return "×"

def get_trunk_score(ratio):
    t = TRUNK_SHIFT_TH
    if ratio < t[0]: return "◎"
    if ratio < t[1]: return "○"
    if ratio < t[2]: return "△"
    return "×"

def _get_side_score(key, val):
    t = SIDE_THRESHOLDS[key]
    if val < t[0]: return "◎"
    if val < t[1]: return "○"
    if val < t[2]: return "△"
    return "×"

def direction(a): return "右下がり" if a >= 0 else "左下がり"
def midpoint(p1, p2): return ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
def cv2pil(img): return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
def pil2cv2(img): return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def draw_text(draw, pos, text, font, color): 
    draw.text(pos, text, font=font, fill=color, stroke_width=1, stroke_fill=color)

def draw_text_center(draw, cx, y, text, font, color):
    bb = draw.textbbox((0, 0), text, font=font)
    tw = bb[2] - bb[0]
    draw.text((cx - tw // 2, y), text, font=font, fill=color, stroke_width=1, stroke_fill=color)

def get_font(size):
    try: return ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB.ttc", size)
    except: return ImageFont.load_default()

CONNECTIONS = [(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]
def px_zoom(lm, w, h, x1, y1, scale):
    return (int((lm.x * w - x1) * scale), int((lm.y * h - y1) * scale))

def draw_skeleton_zoom(img, lm, w, h, x1, y1, scale):
    for s, e in CONNECTIONS:
        if lm[s].visibility > 0.3 and lm[e].visibility > 0.3:
            p1 = px_zoom(lm[s], w, h, x1, y1, scale)
            p2 = px_zoom(lm[e], w, h, x1, y1, scale)
            cv2.line(img, p1, p2, (150,150,150), 1, cv2.LINE_AA)
    for l in lm:
        if l.visibility > 0.3:
            cv2.circle(img, px_zoom(l, w, h, x1, y1, scale), 3, (80,230,120), -1)

def draw_meas_line_zoom(img, lm1, lm2, sc, w, h, x1, y1, scale):
    c = SCORE_BGR[sc]
    p1 = px_zoom(lm1, w, h, x1, y1, scale)
    p2 = px_zoom(lm2, w, h, x1, y1, scale)
    cv2.line(img, p1, p2, c, 2, cv2.LINE_AA)
    cv2.circle(img, p1, 6, c, -1); cv2.circle(img, p2, 6, c, -1)

def draw_midline_zoom(img, lm, w, h, x1, y1, scale):
    lx_orig = (lm[27].x + lm[28].x) / 2 * w
    lx = int((lx_orig - x1) * scale)
    py_top = int((lm[0].y * h - h * 0.1 - y1) * scale)
    py_bot = int((max(lm[27].y, lm[28].y) * h + h * 0.02 - y1) * scale)
    ih = img.shape[0]
    for y in range(max(py_top, 10), min(py_bot, ih), 20):
        cv2.line(img, (lx, y), (lx, min(y+10, py_bot)), MIDLINE_COL_BGR, 2, cv2.LINE_AA)
    
    pts = [
        midpoint(px_zoom(lm[7],w,h,x1,y1,scale),  px_zoom(lm[8],w,h,x1,y1,scale)),
        midpoint(px_zoom(lm[11],w,h,x1,y1,scale), px_zoom(lm[12],w,h,x1,y1,scale)),
        midpoint(px_zoom(lm[23],w,h,x1,y1,scale), px_zoom(lm[24],w,h,x1,y1,scale)),
        (lx, int((max(lm[27].y, lm[28].y)*h - y1)*scale))
    ]
    for pt in pts:
        cv2.circle(img, pt, 6, MIDLINE_COL_BGR, -1)
        cv2.circle(img, pt, 8, (255, 255, 255), 1)
        if pt[0] != lx:
            cv2.arrowedLine(img, (lx, pt[1]), (pt[0], pt[1]), MIDLINE_COL_BGR, 2, cv2.LINE_AA, tipLength=0.2)
    return pts

def _calc_total_score(scores):
    deduction = 0
    mapping = {"◎": 0, "○": 2, "△": 8, "×": 15}
    for s in scores:
        deduction += mapping.get(s, 5)
    return max(0, 100 - deduction)

def calc_body_risks(sc_h, sc_s, sc_p, ts_sc, s_a, p_a):
    pd = "右側" if p_a >= 0 else "左側"; sd = "右肩" if s_a >= 0 else "左肩"
    risks = []
    if sc_p == "×" or ts_sc == "×": risks.append(("足", "△", f"骨盤の傾きで左右バランスが崩れています。{pd}への負担が大きい状態です。"))
    else: risks.append(("足", "◎", "体重分配は良好です。"))
    if sc_p == "×" or ts_sc in ("△", "×"): risks.append(("膝", "△", f"{pd}の膝にストレスが偏りやすい状態です。"))
    else: risks.append(("膝", "◎", "膝へのバランスは良好です。"))
    if sc_p == "×": risks.append(("股関節", "×", f"{pd}の股関節痛のリスクが高い状態です。"))
    elif sc_p == "△": risks.append(("股関節", "△", "骨盤傾きが股関節への負荷につながっています。"))
    else: risks.append(("股関節", "◎", "股関節への負荷は左右均等です。"))
    if sc_p == "×" or ts_sc == "×": risks.append(("腰", "×", "腰椎に強い非対称な負荷がかかっています。"))
    elif sc_p == "△": risks.append(("腰", "△", f"{pd}の腰への負担が片側に偏っています。"))
    else: risks.append(("腰", "◎", "腰椎バランスは良好です。"))
    if sc_s == "×": risks.append(("肩", "×", f"{sd}が大きく下がり、反対側の腱板に負荷がかかっています。"))
    elif sc_s == "△": risks.append(("肩", "△", "肩こりや腕のだるさが出やすい状態です。"))
    else: risks.append(("肩", "◎", "肩のバランスは良好です。"))
    if sc_h == "×": risks.append(("首", "×", "頸椎に偏った負荷がかかり、頭痛やしびれのリスクがあります。"))
    elif sc_h == "△": risks.append(("首", "△", "首の筋肉に左右差が生じやすく、疲れやすい状態です。"))
    else: risks.append(("首", "◎", "首への負荷は左右均等です。"))
    return risks

def calc_side_risks(fhp_sc, rs_sc, trk_sc, pel_sc, fval, rval):
    risks = []
    if fhp_sc == "×": risks.append(("首", "×", f"頭部が大幅に前出し（約{fval:.0f}%）、頸椎負荷が高まっています。"))
    elif fhp_sc == "△": risks.append(("首", "△", "頭部の前方変位により慢性的な首の張りが出やすいです。"))
    else: risks.append(("首", "◎", "首の前後バランスは良好です。"))
    if rs_sc == "×": risks.append(("肩", "×", f"肩が大きく巻き込み（約{rval:.0f}%）、腱板を痛めるリスクがあります。"))
    elif rs_sc == "△": risks.append(("肩", "△", "肩の巻き込みにより肩こりが慢性化しやすいです。"))
    else: risks.append(("肩", "◎", "肩の前後バランスは良好です。"))
    if pel_sc == "×": risks.append(("腰・骨盤", "×", "骨盤の傾きが強く、腰椎への負荷が高い状態です。"))
    else: risks.append(("腰・骨盤", "◎", "骨盤バランスは良好です。"))
    if trk_sc == "×": risks.append(("全身", "×", "重心ラインが大きくずれ、全身が疲れやすい姿勢です。"))
    else: risks.append(("全身", "◎", "全身の垂直バランスは良好です。"))
    return risks

MAX_CHARS = 28
def _measure_panel_height(items, risks): return max(100 + len(items)*72 + len(risks)*45 + 500, 1400)
def _measure_side_panel_height(items, risks): return max(100 + len(items)*72 + len(risks)*45 + 500, 1400)

def calc_future_risks(scores):
    weights = {"◎": 0, "○": 1, "△": 3, "×": 6}
    total_val = sum(weights.get(s, 0) for s in scores)
    max_val = len(scores) * 6
    if max_val == 0: return []
    base_risk = (total_val / max_val) * 100
    risks = [
        {"name": "血流・代謝不全", "val": min(base_risk * 1.1 + 10, 99), "desc": "筋ポンプ作用低下による冷え、むくみの定着"},
        {"name": "自律神経の乱れ", "val": min(base_risk * 0.9 + 5, 99), "desc": "頸椎負荷による不眠・頭痛等の不調リスク"},
        {"name": "内臓圧迫・消化器", "val": min(base_risk * 0.8 + 5, 99), "desc": "前傾姿勢による腹部圧迫と活動効率低下"},
        {"name": "将来的な慢性痛", "val": min(base_risk * 1.3 + 15, 99), "desc": "特定部位への過負荷（ヘルニア・変形性等）"}
    ]
    return risks

def build_panel(items, risks, pw, ih):
    scores = [it["score"] for it in items] + [r[1] for r in risks]
    f_risks = calc_future_risks(scores)
    total_pt = _calc_total_score(scores)
    
    ph = max(_measure_panel_height(items, risks), ih)
    p = Image.new("RGB", (pw, ph), PANEL_BG); dr = ImageDraw.Draw(p)
    fH, fB, fS, fXS, fXXS = get_font(22), get_font(19), get_font(16), get_font(15), get_font(14)
    dr.rectangle([(0,0),(pw,50)], fill=(30,40,70)); draw_text_center(dr, pw//2, 12, "[ AI 姿勢解析：正面観察 ]", fH, WHITE)
    
    y = 65
    dr.rectangle([(10,y),(pw-10,y+60)], fill=(32,38,58), outline=(60,80,150))
    draw_text(dr, (25, y+18), "あなたの姿勢総合スコア", fS, WHITE)
    score_col = (255, 215, 0) if total_pt > 70 else (255, 100, 100)
    draw_text(dr, (pw-120, y+10), str(total_pt), get_font(38), score_col)
    draw_text(dr, (pw-55, y+25), "/ 100 pt", fS, WHITE)
    
    y = 135; dr.rectangle([(10,y),(pw-10,y+20)], fill=(28,42,66), outline=LINE_COL); draw_text(dr, (18,y+4), "正面理想：各ライン水平 0.0°　正中線偏位 0%", fXS, GREEN_IDEAL); y += 38
    draw_text(dr, (18,y), "▌ 計測結果", fH, WHITE); y += 32
    for item in items:
        col = SCORE_RGB[item["score"]]; dr.rectangle([(10,y),(pw-10,y+64)], fill=(34,40,68), outline=LINE_COL)
        draw_text(dr, (18,y+8), item["name"], fB, WHITE); bx = pw-50; dr.ellipse([(bx,y+6),(bx+24,y+30)], fill=col); draw_text(dr, (bx+4,y+10), item["score"], fXS, (10,10,10))
        draw_text(dr, (18,y+36), f"理想: {item['normal']:+.1f}°", fS, GREEN_IDEAL); draw_text(dr, (138,y+36), f"実測: {item['measured']:+.1f}°", fS, YELLOW); draw_text(dr, (278,y+36), f"偏差: {item['diff']:.1f}° ({item['direction']})", fS, col); y += 72
        
    y += 10; dr.line([(10,y),(pw-10,y)], fill=(70,78,120), width=2); y += 15; draw_text(dr, (18,y), "▌ 部位別リスク予測", fH, WHITE); y += 32; draw_text(dr, (20,y), "※現在の姿勢データから推定される傾向です", fXXS, GRAY); y += 25
    for pt, sc, msg in risks:
        col = SCORE_RGB[sc]; dr.line([(10,y),(pw-10,y)], fill=LINE_COL); y += 8
        draw_text(dr, (18,y), f"【{pt}】", fS, WHITE); bx = 125
        dr.ellipse([(bx, y), (bx+22, y+22)], fill=col, outline=(30,30,50), width=1)
        draw_text(dr, (bx+4, y+2), sc, fXXS, (10,10,10))
        rem = msg; wrp = []
        while len(rem) > MAX_CHARS: wrp.append(rem[:MAX_CHARS]); rem = rem[MAX_CHARS:]
        if rem: wrp.append(rem)
        for i, t in enumerate(wrp): draw_text(dr, (bx + 40, y + i*16), t, fXXS, col)
        y += 45

    y += 18; dr.rectangle([(10,y),(pw-10,y+450)], fill=(20,24,40), outline=(255, 80, 80)); y += 18; draw_text(dr, (20,y), ">> 未来の健康リスク：5-10年後予報", get_font(19), (255,100,100)); y += 48
    for fr in f_risks:
        draw_text(dr, (25,y), fr["name"], fB, WHITE); y += 28
        bar_x1, bar_x2 = 25, pw - 85
        dr.rectangle([(bar_x1, y), (bar_x2, y + 12)], fill=(40, 40, 60))
        gw = int((bar_x2 - bar_x1) * (fr["val"] / 100)); gcol = (255, 60, 60) if fr["val"] > 60 else (255, 180, 40)
        dr.rectangle([(bar_x1, y), (bar_x1 + gw, y + 12)], fill=gcol)
        draw_text(dr, (bar_x2 + 10, y - 4), f"{fr['val']:.0f}%", fS, gcol); y += 22
        draw_text(dr, (25,y), f"● {fr['desc']}", fXXS, GRAY); y += 42

    y += 10; dr.line([(10,y),(pw-10,y)], fill=(70,78,120), width=2); y += 12; draw_text(dr, (18,y), "▌ スコア基準", fH, WHITE); y += 22
    leg = [("頭部",[("◎","<1°"),("○","2°"),("△","4°"),("×","4°+")]),("肩",[("◎","<1°"),("○","2°"),("△","3.5°"),("×","3.5°+")]),("骨盤",[("◎","<0.5°"),("○","1.5°"),("△","3°"),("×","3°+")]),("正中線",[("◎","<2%"),("○","4%"),("△","7%"),("×","7%+")])]
    for lb, cr in leg:
        dr.line([(10,y),(pw-10,y)], fill=LINE_COL); y += 6; draw_text(dr, (18,y), f"【{lb}】", fXS, GRAY); ox = 18+80
        for ic, vl in cr: col = SCORE_RGB[ic]; draw_text(dr, (ox,y), ic, fXS, col); draw_text(dr, (ox+20,y), vl, fXXS, col); ox += 105
        y += 24
    y += 10; dr.line([(10,y),(pw-10,y)], fill=LINE_COL); y += 8; draw_text(dr, (18,y), "▌ 評価基準の出典", get_font(13), WHITE); y += 18
    for r in ["・ Kendall et al. (2005)", "・ Magee DJ. (2014)", "・ 日本リハ会 姿勢評価GL"]: draw_text(dr, (18,y), r, fXXS, GRAY); y += 15
    return pil2cv2(np.array(p))

def build_side_panel(items, risks, pw, ih):
    scores = [it["score"] for it in items] + [r[1] for r in risks]
    f_risks = calc_future_risks(scores)
    total_pt = _calc_total_score(scores)

    ph = max(_measure_side_panel_height(items, risks), ih)
    p = Image.new("RGB", (pw, ph), PANEL_BG); dr = ImageDraw.Draw(p)
    fH, fB, fS, fXS, fXXS = get_font(22), get_font(19), get_font(16), get_font(15), get_font(14)
    dr.rectangle([(0,0),(pw,50)], fill=(30,40,70)); draw_text_center(dr, pw//2, 12, "[ AI 姿勢解析：側面観察 ]", fH, WHITE)
    
    y = 65
    dr.rectangle([(10,y),(pw-10,y+60)], fill=(32,38,58), outline=(60,80,150))
    draw_text(dr, (25, y+18), "あなたの姿勢総合スコア", fS, WHITE)
    score_col = (255, 215, 0) if total_pt > 70 else (255, 100, 100)
    draw_text(dr, (pw-120, y+10), str(total_pt), get_font(38), score_col)
    draw_text(dr, (pw-55, y+25), "/ 100 pt", fS, WHITE)
    
    y = 135; dr.rectangle([(10,y),(pw-10,y+20)], fill=(28,42,66), outline=LINE_COL); draw_text(dr, (18,y+4), "側面理想：耳〜足首が一直線", fXS, GREEN_IDEAL); y += 38
    draw_text(dr, (18,y), "▌ 計測結果", fH, WHITE); y += 32
    for item in items:
        col = SCORE_RGB[item["score"]]; dr.rectangle([(10,y),(pw-10,y+64)], fill=(34,40,68), outline=LINE_COL)
        draw_text(dr, (18,y+8), item["name"], fB, WHITE); bx = pw-50; dr.ellipse([(bx,y+6),(bx+24,y+30)], fill=col); draw_text(dr, (bx+4,y+10), item["score"], fXS, (10,10,10))
        draw_text(dr, (18,y+36), f"理想: {item['ideal']}", fS, GREEN_IDEAL); draw_text(dr, (138,y+36), f"実測: {item['measured']}", fS, YELLOW); draw_text(dr, (278,y+36), f"偏差: {item['diff']}", fS, col); y += 72
        
    y += 10; dr.line([(10,y),(pw-10,y)], fill=(70,78,120), width=2); y += 15; draw_text(dr, (18,y), "▌ 部位別リスク予測", fH, WHITE); y += 32; draw_text(dr, (20,y), "※現在の姿勢データから推定される傾向です", fXXS, GRAY); y += 25
    for pt, sc, msg in risks:
        col = SCORE_RGB[sc]; dr.line([(10,y),(pw-10,y)], fill=LINE_COL); y += 8
        draw_text(dr, (18,y), f"【{pt}】", fS, WHITE); bx = 125
        dr.ellipse([(bx, y), (bx+22, y+22)], fill=col, outline=(30,30,50), width=1)
        draw_text(dr, (bx+4, y+2), sc, fXXS, (10,10,10))
        rem = msg; wrp = []
        while len(rem) > MAX_CHARS: wrp.append(rem[:MAX_CHARS]); rem = rem[MAX_CHARS:]
        if len(rem) > 0: wrp.append(rem)
        for i, t in enumerate(wrp): draw_text(dr, (bx+40, y+i*16), t, fXXS, col)
        y += 45

    y += 18; dr.rectangle([(10,y),(pw-10,y+450)], fill=(20,24,40), outline=(255, 80, 80)); y += 18; draw_text(dr, (20,y), ">> 未来の健康リスク：5-10年後予報", get_font(19), (255,100,100)); y += 48
    for fr in f_risks:
        draw_text(dr, (25,y), fr["name"], fB, WHITE); y += 28
        bar_x1, bar_x2 = 25, pw - 85
        dr.rectangle([(bar_x1, y), (bar_x2, y + 12)], fill=(40, 40, 60))
        gw = int((bar_x2 - bar_x1) * (fr["val"] / 100)); gcol = (255, 60, 60) if fr["val"] > 60 else (255, 180, 40)
        dr.rectangle([(bar_x1, y), (bar_x1 + gw, y + 12)], fill=gcol)
        draw_text(dr, (bar_x2 + 10, y - 4), f"{fr['val']:.0f}%", fS, gcol); y += 22
        draw_text(dr, (25,y), f"● {fr['desc']}", fXXS, GRAY); y += 42

    y += 10; dr.line([(10,y),(pw-10,y)], fill=(70,78,120), width=2); y += 12; draw_text(dr, (18,y), "▌ スコア基準", fH, WHITE); y += 22
    leg = [("前方頭位",[("◎","<5%"),("○","10%"),("△","18%"),("×",">18%")]),("ラウンド肩",[("◎","<4%"),("○","8%"),("△","14%"),("×",">14%")]),("骨盤前後傾",[("◎","<2°"),("○","5°"),("△","10°"),("×",">10°")]),("体幹ライン",[("◎","<3%"),("○","6%"),("△","12%"),("×",">12%")])]
    for lb, cr in leg:
        dr.line([(10,y),(pw-10,y)], fill=LINE_COL); y += 6; draw_text(dr, (18,y), f"【{lb}】", fXS, GRAY); ox = 18+115
        for ic, vl in cr: col = SCORE_RGB[ic]; draw_text(dr, (ox,y), ic, fXS, col); draw_text(dr, (ox+20,y), vl, fXXS, col); ox += 100
        y += 24
    y += 10; dr.line([(10,y),(pw-10,y)], fill=LINE_COL); y += 8; draw_text(dr, (18,y), "▌ 評価基準の出典", get_font(13), WHITE); y += 18
    for r in ["・ Kendall et al. (2005)", "・ Magee DJ. (2014)","・ Griegel-Morris P. (1992)", "・ 日本リハ会 姿勢評価GL"]: draw_text(dr, (18,y), r, fXXS, GRAY); y += 15
    return pil2cv2(np.array(p))

def draw_cog_indicator(img, lm, view, w, h, x1, y1, scale):
    p_base = midpoint(px_zoom(lm[27],w,h,x1,y1,scale), px_zoom(lm[28],w,h,x1,y1,scale))
    py = img.shape[0] - 25
    shift = (lm[0].x - (lm[27].x + lm[28].x) / 2) # 簡易
    
    if view == 'front':
        bar_w = 120
        cv2.rectangle(img, (p_base[0]-bar_w//2, py), (p_base[0]+bar_w//2, py+8), (60,60,60), -1)
        cog_x = max(p_base[0]-bar_w//2, min(p_base[0]+bar_w//2, int(p_base[0] + (shift * bar_w / 2))))
        cv2.circle(img, (cog_x, py+4), 6, (0, 255, 255), -1)
    return img

def build_comparison_panel(it1, it2, risks, pw, ih, mode):
    p = Image.new("RGB", (pw, ih), PANEL_BG); dr = ImageDraw.Draw(p)
    fH, fB, fS, fXS, fXXS = get_font(22), get_font(19), get_font(16), get_font(14), get_font(13)
    dr.rectangle([(0,0),(pw,50)], fill=(30,40,70)); draw_text_center(dr, pw//2, 8, f"�� 姿勢変化・改善指標（{mode}）", get_font(28), WHITE)
    y = 65; draw_text(dr, (18,y), "▌ 姿勢データの変化", fH, WHITE); y += 38
    for i, (b, a) in enumerate(zip(it1, it2)):
        dr.rectangle([(10,y),(pw-10,y+78)], fill=(34,40,68), outline=LINE_COL)
        draw_text(dr, (18,y+8), b["n"], fB, WHITE)
        diff = abs(b["v"]) - abs(a["v"]); col_diff = (80,255,150) if diff > 0.1 else (255,100,80)
        draw_text(dr, (pw-140, y+42), f"改善: {diff:+.1f}", fS, col_diff)
        y += 88
    return pil2cv2(np.array(p))

