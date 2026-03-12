"""
AI姿勢解析システム v9 (Performance & Accuracy Enhanced)
-----------------------------------------
・PoseAnalyzerクラス化による高速化（モデルのメモリ保持）
・正面/側面解析の統合
・精度調整（Detection Confidence）
・リサイズ補間（Cubic）による画質維持
"""

import cv2
import mediapipe as mp
import math
import numpy as np
import os
import glob
from PIL import Image, ImageDraw, ImageFont
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ─── 設定・定数 ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FONT_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "NotoSansJP.ttf"),
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]

def get_font(size):
    for path in FONT_CANDIDATES:
        try: return ImageFont.truetype(path, size, index=0)
        except: continue
    return ImageFont.load_default()

THRESHOLDS = {
    "頭部": (1.0, 2.0, 4.0),
    "肩":   (1.0, 2.0, 3.5),
    "骨盤": (0.5, 1.5, 3.0),
}
TRUNK_SHIFT_TH = (0.02, 0.04, 0.07)

SIDE_THRESHOLDS = {
    "FHP":             (0.05, 0.10, 0.18),
    "ラウンドショルダー": (0.04, 0.08, 0.14),
    "体幹ライン":       (0.03, 0.06, 0.12),
    "骨盤前後傾":      (2.0,  5.0, 10.0),
}

SCORE_RGB = {
    "◎": (80, 210, 100), "○": (80, 200, 240), "△": (255, 160, 40), "×": (230, 60, 50),
}
SCORE_BGR = {k: (v[2], v[1], v[0]) for k, v in SCORE_RGB.items()}
PANEL_BG, LINE_COL = (24, 29, 48), (55, 65, 100)
WHITE, GRAY, YELLOW, GREEN_IDEAL = (240, 245, 255), (140, 150, 175), (255, 220, 80), (80, 220, 140)
MIDLINE_COL_BGR, MIDLINE_COL_RGB = (220, 220, 50), (50, 220, 220)

# ─── 解析エンジンクラス ────────────────────────────────────────────────────────
class PoseAnalyzer:
    def __init__(self, model_path):
        print(f"解析エンジン初期化中 (Model: {os.path.basename(model_path)})...")
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        # 精度向上のため confidence を微調整 (デフォルトは0.5)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            min_pose_detection_confidence=0.6,
            min_pose_presence_confidence=0.6
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        print("解析エンジン準備完了。")

    def __del__(self):
        if hasattr(self, 'detector'):
            self.detector.close()

    def analyze(self, image_path, output_path, view_type='auto'):
        """単一画像の解析メインエントリ"""
        img = cv2.imread(image_path)
        if img is None: return None

        # 1. リサイズ（標準化）
        orig_h, orig_w = img.shape[:2]
        target_h = 800
        target_w = int(orig_w * (target_h / orig_h))
        # 高品質補間
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        h, w = target_h, target_w

        # 2. 姿勢検出
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        result = self.detector.detect(mp_image)

        # メモリ解放のヒント（ガベージコレクションを明示的に呼ぶ準備）
        import gc

        if not result.pose_landmarks:
            print(f"  [SKIP] 人物が検出されませんでした: {os.path.basename(image_path)}")
            return None

        lm = result.pose_landmarks[0]

        # 3. 向き判定
        if view_type == 'auto':
            view = self._detect_view(lm)
        else:
            view = view_type

        # 後処理
        res = None
        if view == 'front':
            res = self._analyze_front(img, lm, w, h, output_path)
        else:
            res = self._analyze_side(img, lm, w, h, output_path)
            
        # 明示的なメモリ解放
        del mp_image, result, img
        gc.collect()
        return res

    def _detect_view(self, lm):
        shldr_dx = abs(lm[11].x - lm[12].x)
        nose_to_leye = abs(lm[0].x - lm[2].x)
        return "front" if shldr_dx > nose_to_leye * 1.5 else "side"

    # ── 内部解析ロジック (正面) ──
    def _analyze_front(self, img, lm, w, h, output_path):
        draw_skeleton(img, lm, w, h)
        head_a   = calc_angle(lm[7],  lm[8],  w, h)
        shldr_a  = calc_angle(lm[11], lm[12], w, h)
        pelvis_a = calc_angle(lm[23], lm[24], w, h)

        sc_head   = get_score("頭部",  abs(head_a))
        sc_shldr  = get_score("肩",    abs(shldr_a))
        sc_pelvis = get_score("骨盤",  abs(pelvis_a))

        draw_meas_line(img, pxcoord(lm[7],w,h),  pxcoord(lm[8],w,h),  sc_head)
        draw_meas_line(img, pxcoord(lm[11],w,h), pxcoord(lm[12],w,h), sc_shldr)
        draw_meas_line(img, pxcoord(lm[23],w,h), pxcoord(lm[24],w,h), sc_pelvis)

        # 基準点：両足首の中点
        foot_mid_x = (lm[27].x + lm[28].x) / 2
        ear_mid_x, shldr_mid_x, pelv_mid_x = (lm[7].x+lm[8].x)/2, (lm[11].x+lm[12].x)/2, (lm[23].x+lm[24].x)/2
        
        # 骨盤幅を基準スケールとする
        pelv_width = max(abs(lm[23].x - lm[24].x), 1e-6)
        
        # 足元（正中線）からのズレを算出
        ear_shift_pct   = (ear_mid_x - foot_mid_x) / pelv_width * 100
        shldr_shift_pct = (shldr_mid_x - foot_mid_x) / pelv_width * 100
        pelv_shift_pct  = (pelv_mid_x - foot_mid_x) / pelv_width * 100
        
        max_shift_abs   = max(abs(ear_shift_pct), abs(shldr_shift_pct), abs(pelv_shift_pct))
        ts_score = get_trunk_score(max_shift_abs / 100)

        draw_midline(img, lm, w, h)
        
        # 写真へのラベル描画
        pil_photo = cv2pil(img)
        dp = ImageDraw.Draw(pil_photo); fl = get_font(16)
        def lbl_near(lm_a, lm_b, text, sc):
            mx, my = int((lm_a.x+lm_b.x)/2*w), int((lm_a.y+lm_b.y)/2*h)-14
            col = SCORE_RGB[sc]; bb = dp.textbbox((0,0), text, font=fl); tw = bb[2]-bb[0]
            dp.rectangle([(mx-tw//2-4,my-2),(mx+tw//2+4,my+14)], fill=(8,12,25))
            dp.text((mx-tw//2, my), text, font=fl, fill=col)
        
        lbl_near(lm[7],  lm[8],  f"頭 {sc_head} {abs(head_a):.1f}°", sc_head)
        lbl_near(lm[11], lm[12], f"肩 {sc_shldr} {abs(shldr_a):.1f}°", sc_shldr)
        lbl_near(lm[23], lm[24], f"盤 {sc_pelvis} {abs(pelvis_a):.1f}°", sc_pelvis)
        
        # 正中線ラベルと偏位
        lx = int(foot_mid_x*w)
        foot_py = h - 20 # 足元ラベル位置
        dp.text((lx+6, foot_py-15), "正中線", font=get_font(11), fill=MIDLINE_COL_RGB)
        
        for mid_x, lm_y, text in [
            (ear_mid_x, lm[7].y, f"{'←' if (ear_mid_x-foot_mid_x)<0 else ''}{abs(ear_shift_pct):.1f}%{'→' if (ear_mid_x-foot_mid_x)>=0 else ''}"),
            (shldr_mid_x, lm[11].y, f"{'←' if (shldr_mid_x-foot_mid_x)<0 else ''}{abs(shldr_shift_pct):.1f}%{'→' if (shldr_mid_x-foot_mid_x)>=0 else ''}"),
            (pelv_mid_x, lm[23].y, f"{'←' if (pelv_mid_x-foot_mid_x)<0 else ''}{abs(pelv_shift_pct):.1f}%{'→' if (pelv_mid_x-foot_mid_x)>=0 else ''}")
        ]:
            px, py = int(mid_x*w), int(lm_y*h)
            dp.text((px+8, py-4), text, font=get_font(14), fill=MIDLINE_COL_RGB)
        
        img = pil2cv2(np.array(pil_photo))
        risk_msgs = calc_body_risks(sc_head, sc_shldr, sc_pelvis, ts_score, shldr_a, pelvis_a)
        score_items = [
            {"name": "頭部（耳の傾き）", "normal": 0.0, "measured": head_a, "diff": abs(head_a), "direction": direction(head_a), "score": sc_head},
            {"name": "肩ライン（傾き）", "normal": 0.0, "measured": shldr_a, "diff": abs(shldr_a), "direction": direction(shldr_a), "score": sc_shldr},
            {"name": "骨盤ライン（傾き）", "normal": 0.0, "measured": pelvis_a, "diff": abs(pelvis_a), "direction": direction(pelvis_a), "score": sc_pelvis},
            {"name": "頭部ズレ（正中線）", "normal": 0.0, "measured": ear_shift_pct, "diff": abs(ear_shift_pct), "direction": "右偏位" if ear_shift_pct > 0 else "左偏位", "score": ts_score},
            {"name": "肩部ズレ（正中線）", "normal": 0.0, "measured": shldr_shift_pct, "diff": abs(shldr_shift_pct), "direction": "右偏位" if shldr_shift_pct > 0 else "左偏位", "score": ts_score},
            {"name": "骨盤ズレ（正中線）", "normal": 0.0, "measured": pelv_shift_pct, "diff": abs(pelv_shift_pct), "direction": "右偏位" if pelv_shift_pct > 0 else "左偏位", "score": ts_score},
        ]
        panel = build_panel(score_items, risk_msgs, 560, h)
        return self._save_final_report(img, panel, output_path, "正面")

    # ── 内部解析ロジック (側面) ──
    def _analyze_side(self, img, lm, w, h, output_path):
        # 向き判定の改善: 口の中点が耳の中点より右なら右向き
        mouth_x = (lm[9].x + lm[10].x) / 2
        ear_avg_x = (lm[7].x + lm[8].x) / 2
        facing_right = mouth_x > ear_avg_x
        
        # 側面では、鼻からX軸上で最も遠い（後方にある）方の耳を「可視耳」として選択する
        # これにより、横を向いた際に顔の内側に予測された誤ったランドマークを回避できる
        idx_ear = 7 if abs(lm[7].x - lm[0].x) > abs(lm[8].x - lm[0].x) else 8
            
        # 選択した耳に合わせて他の指標セット（肩、腰、膝、足首）を定義
        idx = [idx_ear, 12, 24, 26, 28] if idx_ear == 8 else [idx_ear, 11, 23, 25, 27]
        ear, shldr, hip, knee, ankle = [lm[i] for i in idx]
        
        ref_len = max(abs(shldr.y - hip.y) * h, 1)
        ear_px_orig = pxcoord(ear,w,h)
        # 耳垂（耳たぶ）の位置に調整: もみあげを避け、確実に耳たぶの位置へ移動
        y_off = int(ref_len * 0.07)
        x_off = int(ref_len * 0.04) * (1 if not facing_right else -1)
        ear_px = (ear_px_orig[0] + x_off, ear_px_orig[1] + y_off)
        shldr_px, hip_px, ankle_px = pxcoord(shldr,w,h), pxcoord(hip,w,h), pxcoord(ankle,w,h)

        fhp_pct = (ear_px[0]/w - shldr.x) * w / ref_len * 100 * (1 if facing_right else -1)
        rs_pct  = (shldr.x - hip.x) * w / ref_len * 100 * (1 if facing_right else -1)
        pel_a   = math.degrees(math.atan2((knee.x-hip.x)*w, (knee.y-hip.y)*h)) * (1 if facing_right else -1)
        trunk_pct = (ear_px[0]/w - ankle.x) * w / ref_len * 100 * (1 if facing_right else -1)

        fhp_sc, rs_sc, pel_sc, trk_sc = [_get_side_score(k, abs(v)/100 if "pct" in k else abs(v)) for k, v in zip(["FHP","ラウンドショルダー","骨盤前後傾","体幹ライン"], [fhp_pct, rs_pct, pel_a, trunk_pct])]
        
        draw_skeleton(img, lm, w, h)
        
        # 垂直基準線（重心線）を描画: 足首から垂直に伸ばす
        ankle_top = (ankle_px[0], max(ear_px[1] - 20, 5))
        for yy in range(ankle_top[1], ankle_px[1], 20):
            cv2.line(img, (ankle_px[0], yy), (ankle_px[0], min(yy+10, ankle_px[1])),
                     MIDLINE_COL_BGR, 2, cv2.LINE_AA)

        pts = [ear_px, shldr_px, hip_px, ankle_px]
        for i in range(len(pts)-1): cv2.line(img, pts[i], pts[i+1], (200,200,50), 2, cv2.LINE_AA)
        for pt in pts: cv2.circle(img, pt, 6, (200,200,50), -1)
        
        # 水平偏差矢印
        def draw_h_diff(p_a, p_b, sc):
            col = SCORE_BGR[sc]; my = (p_a[1]+p_b[1])//2
            cv2.arrowedLine(img, (p_b[0],my), (p_a[0],my), col, 2, cv2.LINE_AA, tipLength=0.2)
            cv2.circle(img, p_a, 7, col, -1); cv2.circle(img, p_b, 7, col, -1)
        draw_h_diff(ear_px, shldr_px, fhp_sc)

        pil_p = cv2pil(img); dp = ImageDraw.Draw(pil_p); fl = get_font(16)
        def s_lbl(pt, txt, sc, dy=-16):
            col = SCORE_RGB[sc]; tx, ty = pt[0]+8, pt[1]+dy
            bb = dp.textbbox((0,0),txt,font=fl); tw = bb[2]-bb[0]
            dp.rectangle([(tx-3,ty-2),(tx+tw+3,ty+14)], fill=(8,12,25))
            dp.text((tx,ty), txt, font=fl, fill=col)
        s_lbl(ear_px, f"耳垂 FHP:{abs(fhp_pct):.0f}% {fhp_sc}", fhp_sc)
        s_lbl(shldr_px, f"肩 RS:{abs(rs_pct):.0f}% {rs_sc}", rs_sc)
        s_lbl(hip_px, f"股 骨盤:{abs(pel_a):.1f}° {pel_sc}", pel_sc, dy=4)
        s_lbl(ankle_px, f"足 体幹:{abs(trunk_pct):.0f}% {trk_sc}", trk_sc, dy=4)
        
        img = pil2cv2(np.array(pil_p))
        risk_msgs = calc_side_risks(fhp_sc, rs_sc, trk_sc, pel_sc, abs(fhp_pct), abs(rs_pct))
        score_items = [
            {"name":"前方頭位(FHP)","ideal":"0%","measured":f"{fhp_pct:+.1f}%","diff":f"{abs(fhp_pct):.1f}%","score":fhp_sc},
            {"name":"ラウンドショルダー","ideal":"0%","measured":f"{rs_pct:+.1f}%","diff":f"{abs(rs_pct):.1f}%","score":rs_sc},
            {"name":"骨盤前後傾","ideal":"0°","measured":f"{pel_a:+.1f}°","diff":f"{abs(pel_a):.1f}°","score":pel_sc},
            {"name":"体幹ライン領域","ideal":"0%","measured":f"{trunk_pct:+.1f}%","diff":f"{abs(trunk_pct):.1f}%","score":trk_sc},
        ]
        panel = build_side_panel(score_items, risk_msgs, 560, h)
        return self._save_final_report(img, panel, output_path, f"側面：{'右向き' if facing_right else '左向き'}")

    def _save_final_report(self, img, panel, output_path, title_suffix):
        # 統合
        ph = panel.shape[0]
        if img.shape[0] < ph:
            pad = np.zeros((ph - img.shape[0], img.shape[1], 3), dtype=np.uint8)
            pad[:] = (18, 22, 35); img = np.vstack([img, pad])
        elif panel.shape[0] < img.shape[0]:
            pad = np.zeros((img.shape[0] - ph, panel.shape[1], 3), dtype=np.uint8)
            pad[:] = (24, 29, 48); panel = np.vstack([panel, pad])
        
        canvas = np.hstack([img, panel])
        bar_h = 52; fw, fh = canvas.shape[1], canvas.shape[0] + bar_h
        full_p = Image.new("RGB", (fw, fh), (28, 34, 58))
        full_d = ImageDraw.Draw(full_p)
        draw_text_center(full_d, fw//2, 8, f"整体院 導 ｜ AI 姿勢解析レポート（{title_suffix}）", get_font(30), WHITE)
        draw_text_center(full_d, fw//2, 32, "MediaPipe Pose Estimation + エビデンスベース解析", get_font(17), GRAY)
        full_p.paste(cv2pil(canvas), (0, bar_h))
        final = pil2cv2(np.array(full_p))
        cv2.imwrite(output_path, final)
        print(f"  ✅ 保存: {os.path.basename(output_path)}")
        return True

# ─── 旧関数互換レイヤー（シングル実行用） ────────────────────────────────────
def analyze_posture(image_path, model_path, output_path):
    engine = PoseAnalyzer(model_path)
    return engine.analyze(image_path, output_path, 'front')

def analyze_posture_side(image_path, model_path, output_path):
    engine = PoseAnalyzer(model_path)
    return engine.analyze(image_path, output_path, 'side')

# ─── 共通ヘルパー (既存のものをコピー) ───────────────────────────────────────
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
def pxcoord(lm, w, h): return (int(lm.x*w), int(lm.y*h))
def midpoint(p1, p2): return ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
def cv2pil(img): return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
def pil2cv2(img): return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
def draw_text(draw, pos, text, font, color): draw.text(pos, text, font=font, fill=color)
def draw_text_center(draw, cx, y, text, font, color):
    bb = draw.textbbox((0, 0), text, font=font); tw = bb[2] - bb[0]
    draw.text((cx - tw // 2, y), text, font=font, fill=color)

CONNECTIONS = [(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]
def draw_skeleton(img, lm, w, h):
    for s, e in CONNECTIONS:
        if lm[s].visibility > 0.3 and lm[e].visibility > 0.3:
            cv2.line(img, pxcoord(lm[s],w,h), pxcoord(lm[e],w,h), (150,150,150), 1, cv2.LINE_AA)
    for l in lm: cv2.circle(img, (int(l.x*w), int(l.y*h)), 3, (80,230,120), -1)

def draw_meas_line(img, p1, p2, sc):
    c = SCORE_BGR[sc]; cv2.line(img, p1, p2, c, 3, cv2.LINE_AA); cv2.circle(img, p1, 7, c, -1); cv2.circle(img, p2, 7, c, -1)

def draw_midline(img, lm, w, h):
    # 正面正中線の下端：左右の足首(27, 28)の中点
    lx = int((lm[27].x + lm[28].x) / 2 * w)
    
    # 描画範囲：頭頂部付近から足元まで
    py_top = int(lm[0].y * h - h * 0.1)
    py_bot = int(max(lm[27].y, lm[28].y) * h + h * 0.02)
    
    # 点線の描画
    for y in range(max(py_top, 10), py_bot, 16):
        cv2.line(img, (lx, y), (lx, min(y+8, py_bot)), MIDLINE_COL_BGR, 2, cv2.LINE_AA)
    
    # 指標ポイント（頭部中点、肩中点、骨盤中点、足元中点）の描画と矢印
    pts = [
        midpoint(pxcoord(lm[7],w,h),  pxcoord(lm[8],w,h)),
        midpoint(pxcoord(lm[11],w,h), pxcoord(lm[12],w,h)),
        midpoint(pxcoord(lm[23],w,h), pxcoord(lm[24],w,h)),
        (lx, int(max(lm[27].y, lm[28].y)*h))
    ]
    for pt in pts:
        cv2.circle(img, pt, 6, MIDLINE_COL_BGR, -1); cv2.circle(img, pt, 8, (255, 255, 255), 1)
        if pt[0] != lx:
            cv2.arrowedLine(img, (lx, pt[1]), (pt[0], pt[1]), MIDLINE_COL_BGR, 2, cv2.LINE_AA, tipLength=0.25)

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

def _calc_risk_row_h(msg): return 38 if len(msg) > MAX_CHARS else 24
MAX_CHARS = 28
def _measure_panel_height(items, risks): return max(100 + len(items)*72 + len(risks)*40 + 350, 1400)
def _measure_side_panel_height(items, risks): return max(100 + len(items)*72 + len(risks)*40 + 350, 1400)

def build_panel(items, risks, pw, ih):
    ph = max(_measure_panel_height(items, risks), ih)
    p = Image.new("RGB", (pw, ph), PANEL_BG); dr = ImageDraw.Draw(p); fT, fH, fB, fS, fXS, fXXS = get_font(28), get_font(22), get_font(19), get_font(16), get_font(15), get_font(14)
    dr.rectangle([(0,0),(pw,50)], fill=(30,40,70)); draw_text_center(dr, pw//2, 8, "🩺 AI 姿勢解析レポート（正面観察）", fT, WHITE)
    y = 58; dr.rectangle([(10,y),(pw-10,y+20)], fill=(28,42,66), outline=LINE_COL); draw_text(dr, (18,y+4), "正面理想：各ライン水平 0.0°　正中線偏位 0%", fXS, GREEN_IDEAL); y += 38
    draw_text(dr, (18,y), "▌ 計測結果", fH, WHITE); y += 32
    for item in items:
        col = SCORE_RGB[item["score"]]; dr.rectangle([(10,y),(pw-10,y+64)], fill=(34,40,68), outline=LINE_COL)
        draw_text(dr, (18,y+8), item["name"], fB, WHITE); bx = pw-50; dr.ellipse([(bx,y+6),(bx+24,y+30)], fill=col); draw_text(dr, (bx+4,y+10), item["score"], fXS, (10,10,10))
        draw_text(dr, (18,y+36), f"理想: {item['normal']:+.1f}°", fS, GREEN_IDEAL); draw_text(dr, (138,y+36), f"実測: {item['measured']:+.1f}°", fS, YELLOW); draw_text(dr, (278,y+36), f"偏差: {item['diff']:.1f}° ({item['direction']})", fS, col); y += 72
    y += 10; dr.line([(10,y),(pw-10,y)], fill=(70,78,120), width=2); y += 12; draw_text(dr, (18,y), "▌ 部位別リスク予測", fH, WHITE); y += 22; draw_text(dr, (18,y), "※姿勢データからの推定です", fXXS, GRAY); y += 20
    for pt, sc, msg in risks:
        col = SCORE_RGB[sc]; dr.line([(10,y),(pw-10,y)], fill=LINE_COL); y += 8; draw_text(dr, (18,y), f"【{pt}】", fS, WHITE); bx = 18+70; dr.ellipse([(bx,y-2),(bx+20,y+18)], fill=col); draw_text(dr, (bx+3,y-1), sc, fXXS, (10,10,10))
        rem = msg; wrp = []
        while len(rem) > MAX_CHARS: wrp.append(rem[:MAX_CHARS]); rem = rem[MAX_CHARS:]
        if rem: wrp.append(rem)
        for i, t in enumerate(wrp): draw_text(dr, (18+105, y+i*15), t, fXXS, col)
        y += 40
    # 凡例・出典
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
    ph = max(_measure_side_panel_height(items, risks), ih)
    p = Image.new("RGB", (pw, ph), PANEL_BG); dr = ImageDraw.Draw(p); fT, fH, fB, fS, fXS, fXXS = get_font(28), get_font(22), get_font(19), get_font(16), get_font(15), get_font(14)
    dr.rectangle([(0,0),(pw,50)], fill=(30,40,70)); draw_text_center(dr, pw//2, 8, "🩺 AI 姿勢解析レポート（側面観察）", fT, WHITE)
    y = 58; dr.rectangle([(10,y),(pw-10,y+20)], fill=(28,42,66), outline=LINE_COL); draw_text(dr, (18,y+4), "側面理想：耳〜足首が一直線", fXS, GREEN_IDEAL); y += 38
    draw_text(dr, (18,y), "▌ 計測結果", fH, WHITE); y += 32
    for item in items:
        col = SCORE_RGB[item["score"]]; dr.rectangle([(10,y),(pw-10,y+64)], fill=(34,40,68), outline=LINE_COL)
        draw_text(dr, (18,y+8), item["name"], fB, WHITE); bx = pw-50; dr.ellipse([(bx,y+6),(bx+24,y+30)], fill=col); draw_text(dr, (bx+4,y+10), item["score"], fXS, (10,10,10))
        draw_text(dr, (18,y+36), f"理想: {item['ideal']}", fS, GREEN_IDEAL); draw_text(dr, (138,y+36), f"実測: {item['measured']}", fS, YELLOW); draw_text(dr, (278,y+36), f"偏差: {item['diff']}", fS, col); y += 72
    y += 10; dr.line([(10,y),(pw-10,y)], fill=(70,78,120), width=2); y += 12; draw_text(dr, (18,y), "▌ 部位別リスク予測", fH, WHITE); y += 22; draw_text(dr, (18,y), "※姿勢データからの推定です", fXXS, GRAY); y += 20
    for pt, sc, msg in risks:
        col = SCORE_RGB[sc]; dr.line([(10,y),(pw-10,y)], fill=LINE_COL); y += 8; draw_text(dr, (18,y), f"【{pt}】", fS, WHITE); bx = 18+70; dr.ellipse([(bx,y-2),(bx+20,y+18)], fill=col); draw_text(dr, (bx+3,y-1), sc, fXXS, (10,10,10))
        rem = msg; wrp = []
        while len(rem) > MAX_CHARS: wrp.append(rem[:MAX_CHARS]); rem = rem[MAX_CHARS:]
        if rem: wrp.append(rem)
        for i, t in enumerate(wrp): draw_text(dr, (18+105, y+i*15), t, fXXS, col)
        y += 40
    # 凡例・出典
    y += 10; dr.line([(10,y),(pw-10,y)], fill=(70,78,120), width=2); y += 12; draw_text(dr, (18,y), "▌ スコア基準", fH, WHITE); y += 22
    leg = [("前方頭位",[("◎","<5%"),("○","10%"),("△","18%"),("×",">18%")]),("ラウンド肩",[("◎","<4%"),("○","8%"),("△","14%"),("×",">14%")]),("骨盤前後傾",[("◎","<2°"),("○","5°"),("△","10°"),("×",">10°")]),("体幹ライン",[("◎","<3%"),("○","6%"),("△","12%"),("×",">12%")])]
    for lb, cr in leg:
        dr.line([(10,y),(pw-10,y)], fill=LINE_COL); y += 6; draw_text(dr, (18,y), f"【{lb}】", fXS, GRAY); ox = 18+115
        for ic, vl in cr: col = SCORE_RGB[ic]; draw_text(dr, (ox,y), ic, fXS, col); draw_text(dr, (ox+20,y), vl, fXXS, col); ox += 100
        y += 24
    y += 10; dr.line([(10,y),(pw-10,y)], fill=LINE_COL); y += 8; draw_text(dr, (18,y), "▌ 評価基準の出典", get_font(13), WHITE); y += 18
    for r in ["・ Kendall et al. (2005)", "・ Magee DJ. (2014)","・ Griegel-Morris P. (1992)", "・ 日本リハ会 姿勢評価GL"]: draw_text(dr, (18,y), r, fXXS, GRAY); y += 15
    return pil2cv2(np.array(p))

if __name__ == "__main__":
    TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(TEST_DIR, "pose_landmarker.task")
    engine = PoseAnalyzer(MODEL_PATH)
    targets = glob.glob(os.path.join(TEST_DIR, "*.jpg"))
    for t in targets:
        if "annotated" in t: continue
        engine.analyze(t, os.path.join(TEST_DIR, f"annotated_v9_{os.path.basename(t)}"))
