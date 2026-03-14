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
        if img is None: return {'success': False, 'error': 'Image load failed'}

        # 1. リサイズ（標準化）
        orig_h, orig_w = img.shape[:2]
        target_h = 600
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
            return {'success': False, 'error': 'No person detected'}

        lm = result.pose_landmarks[0]

        # 3. 向き判定
        if view_type == 'auto':
            view = self._detect_view(lm)
        else:
            view = view_type

        # 後処理
        res = None
        if view == 'front':
            success, data = self._analyze_front(img, lm, w, h, output_path)
            res = {'success': success, 'data': data, 'view': 'front'}
        else:
            success, data = self._analyze_side(img, lm, w, h, output_path)
            res = {'success': success, 'data': data, 'view': 'side'}
            
        # 明示的なメモリ解放
        del mp_image, result, img
        gc.collect()
        return res

    def analyze_comparison(self, img_path1, img_path2, output_path, view_type='auto'):
        """2枚の画像を比較解析する"""
        import gc
        img1_orig, img2_orig = cv2.imread(img_path1), cv2.imread(img_path2)
        if img1_orig is None or img2_orig is None: return {'success': False, 'error': 'Image load failed'}

        # 解析用リサイズ
        def prep(im):
            h, w = im.shape[:2]; th = 600; tw = int(w * (th/h))
            return cv2.resize(im, (tw, th), interpolation=cv2.INTER_CUBIC)
        i1, i2 = prep(img1_orig), prep(img2_orig)

        # 姿勢検出
        def detect(im):
            mpi = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            res = self.detector.detect(mpi)
            return mpi, res
        mpi1, res1 = detect(i1); mpi2, res2 = detect(i2)

        if not res1.pose_landmarks or not res2.pose_landmarks:
            print(f"  [SKIP] 比較対象の人物が検出されませんでした")
            return {'success': False, 'error': 'No person detected in one or both images'}

        lm1, lm2 = res1.pose_landmarks[0], res2.pose_landmarks[0]
        view = view_type if view_type != 'auto' else self._detect_view(lm2)

        # 個別解析と描画済みイメージの取得
        if view == 'front':
            res_img1, items1, risks1 = self._process_front_for_comp(i1, lm1)
            res_img2, items2, risks2 = self._process_front_for_comp(i2, lm2)
            panel = build_comparison_panel(items1, items2, risks2, 560, 1400, "正面")
        else:
            res_img1, items1, risks1 = self._process_side_for_comp(i1, lm1)
            res_img2, items2, risks2 = self._process_side_for_comp(i2, lm2)
            panel = build_comparison_panel(items1, items2, risks2, 560, 1400, "側面")

        # レポート合成 [Before][After][Panel]
        success = self._save_comparison_report(res_img1, res_img2, panel, output_path, f"比較（{view}）")

        # 数値データを辞書で返す
        data = {
            'before': items1,
            'after': items2,
            'view': view
        }

        del mpi1, res1, mpi2, res2, i1, i2, img1_orig, img2_orig
        gc.collect()
        return {'success': success, 'data': data}

    def _process_front_for_comp(self, img, lm):
        """比較用の正面描画（内部処理）"""
        h, w = img.shape[:2]
        head_a = calc_angle(lm[7], lm[8], w, h); shldr_a = calc_angle(lm[11], lm[12], w, h); pelvis_a = calc_angle(lm[23], lm[24], w, h)
        sc_h, sc_s, sc_p = get_score("頭部", abs(head_a)), get_score("肩", abs(shldr_a)), get_score("骨盤", abs(pelvis_a))
        
        f_mx = (lm[27].x+lm[28].x)/2; p_w = max(abs(lm[23].x-lm[24].x), 1e-6)
        e_s, s_s, p_s = (lm[7].x+lm[8].x)/2-f_mx, (lm[11].x+lm[12].x)/2-f_mx, (lm[23].x+lm[24].x)/2-f_mx
        e_pct, s_pct, p_pct = e_s/p_w*100, s_s/p_w*100, p_s/p_w*100
        ts_sc = get_trunk_score(max(abs(e_pct), abs(s_pct), abs(p_pct))/100)

        x1, y1, x2, y2 = self._get_crop_box(lm, w, h); img_c = img[y1:y2, x1:x2]
        target_ph = 1400; scale = target_ph / img_c.shape[0]
        img_f = cv2.resize(img_c, (int(img_c.shape[1]*scale), target_ph), interpolation=cv2.INTER_CUBIC)
        
        draw_skeleton_zoom(img_f, lm, w, h, x1, y1, scale)
        draw_meas_line_zoom(img_f, lm[11], lm[12], sc_s, w, h, x1, y1, scale)
        draw_meas_line_zoom(img_f, lm[23], lm[24], sc_p, w, h, x1, y1, scale)
        draw_midline_zoom(img_f, lm, w, h, x1, y1, scale)

        items = [
            {"n":"頭部傾き","v":head_a,"s":sc_h}, {"n":"肩傾き","v":shldr_a,"s":sc_s}, {"n":"骨盤傾き","v":pelvis_a,"s":sc_p},
            {"n":"頭部ズレ","v":e_pct,"s":ts_sc}, {"n":"肩部ズレ","v":s_pct,"s":ts_sc}, {"n":"骨盤ズレ","v":p_pct,"s":ts_sc}
        ]
        risks = calc_body_risks(sc_h, sc_s, sc_p, ts_sc, shldr_a, pelvis_a)
        return img_f, items, risks

    def _process_side_for_comp(self, img, lm):
        """比較用の側面描画（内部処理）"""
        h, w = img.shape[:2]; mx, ex = (lm[9].x+lm[10].x)/2, (lm[7].x+lm[8].x)/2
        fr = mx > ex; ie = 7 if abs(lm[7].x-lm[0].x)>abs(lm[8].x-lm[0].x) else 8
        idx = [ie, 12, 24, 26, 28] if ie==8 else [ie, 11, 23, 25, 27]
        ear, shldr, hip, knee, ankle = [lm[i] for i in idx]; ref = max(abs(shldr.y-hip.y)*h, 1)
        xo = int(ref*0.04)*(1 if not fr else -1); yo = int(ref*0.07)
        ear_px_o = (int(ear.x*w+xo), int(ear.y*h+yo))
        f_p, r_p = (ear_px_o[0]/w-shldr.x)*w/ref*100*(1 if fr else -1), (shldr.x-hip.x)*w/ref*100*(1 if fr else -1)
        p_a = math.degrees(math.atan2((knee.x-hip.x)*w, (knee.y-hip.y)*h))*(1 if fr else -1)
        t_p = (ear_px_o[0]/w-ankle.x)*w/ref*100*(1 if fr else -1)
        f_s, r_s, p_s, t_s = [_get_side_score(k, abs(v)/(100 if "p" in k else 1)) for k,v in zip(["FHP","ラウンドショルダー","骨盤前後傾","体幹ライン"],[f_p,r_p,p_a,t_p])]

        x1, y1, x2, y2 = self._get_crop_box(lm, w, h); img_c = img[y1:y2, x1:x2]
        target_ph = 1400; scale = target_ph / img_c.shape[0]; img_f = cv2.resize(img_c, (int(img_c.shape[1]*scale), target_ph), interpolation=cv2.INTER_CUBIC)
        
        draw_skeleton_zoom(img_f, lm, w, h, x1, y1, scale)
        ax = px_zoom(ankle,w,h,x1,y1,scale); ex_p = (int((ear_px_o[0]-x1)*scale), int((ear_px_o[1]-y1)*scale))
        sx = px_zoom(shldr,w,h,x1,y1,scale); hx = px_zoom(hip,w,h,x1,y1,scale)
        for yy in range(max(ex_p[1]-50,20), ax[1], 25): cv2.line(img_f, (ax[0],yy), (ax[0],min(yy+12,ax[1])), MIDLINE_COL_BGR, 2, cv2.LINE_AA)
        pts = [ex_p, sx, hx, ax]
        for i in range(len(pts)-1): cv2.line(img_f, pts[i], pts[i+1], (200,200,50), 2, cv2.LINE_AA)
        for p in pts: cv2.circle(img_f, p, 7, (200,200,50), -1)

        items = [{"n":"FHP","v":f_p,"s":f_s}, {"n":"ラウンド肩","v":r_p,"s":r_s}, {"n":"骨盤前後傾","v":p_a,"s":p_s}, {"n":"体幹領域","v":t_p,"s":t_s}]
        risks = calc_side_risks(f_s,r_s,t_s,p_s,abs(f_p),abs(r_p))
        return img_f, items, risks

    def _save_comparison_report(self, img1, img2, panel, output_path, title):
        # [Before][After][Panel]
        # img1, img2 の横幅を揃える（アスペクト比維持のため最大幅に合わせる必要はないが、レイアウトとして整える）
        canvas = np.hstack([img1, img2, panel])
        bar_h = 60; fw, fh = canvas.shape[1], canvas.shape[0] + bar_h
        full_p = Image.new("RGB", (fw, fh), (28, 34, 58)); draw = ImageDraw.Draw(full_p)
        draw_text_center(draw, fw//2, 10, f"整体院 導 ｜ AI 姿勢比較レポート（{title}）", get_font(34), WHITE)
        
        # Before/After のラベル
        w1 = img1.shape[1]; w2 = img2.shape[1]
        draw.rectangle([(0, bar_h), (w1, bar_h+40)], fill=(40,50,90))
        draw_text_center(draw, w1//2, bar_h+8, "【 BEFORE 】", get_font(24), YELLOW)
        draw.rectangle([(w1, bar_h), (w1+w2, bar_h+40)], fill=(50,70,120))
        draw_text_center(draw, w1 + w2//2, bar_h+8, "【 AFTER 】", get_font(24), (80,255,150))
        
        full_p.paste(cv2pil(canvas), (0, bar_h))
        cv2.imwrite(output_path, pil2cv2(np.array(full_p)))
        return True

    def _detect_view(self, lm):
        shldr_dx = abs(lm[11].x - lm[12].x)
        nose_to_leye = abs(lm[0].x - lm[2].x)
        return "front" if shldr_dx > nose_to_leye * 1.5 else "side"

    # ── 内部解析ロジック (正面) ──
    def _analyze_front(self, img, lm, w, h, output_path):
        # 1. 解析（数値計算のみ先に実施）
        head_a   = calc_angle(lm[7],  lm[8],  w, h)
        shldr_a  = calc_angle(lm[11], lm[12], w, h)
        pelvis_a = calc_angle(lm[23], lm[24], w, h)
        sc_head, sc_shldr, sc_pelvis = get_score("頭部", abs(head_a)), get_score("肩", abs(shldr_a)), get_score("骨盤", abs(pelvis_a))

        foot_mid_x = (lm[27].x + lm[28].x) / 2
        ear_mid_x, shldr_mid_x, pelv_mid_x = (lm[7].x+lm[8].x)/2, (lm[11].x+lm[12].x)/2, (lm[23].x+lm[24].x)/2
        pelv_width = max(abs(lm[23].x - lm[24].x), 1e-6)
        ear_shift_pct, shldr_shift_pct, pelv_shift_pct = (ear_mid_x - foot_mid_x) / pelv_width * 100, (shldr_mid_x - foot_mid_x) / pelv_width * 100, (pelv_mid_x - foot_mid_x) / pelv_width * 100
        ts_score = get_trunk_score(max(abs(ear_shift_pct), abs(shldr_shift_pct), abs(pelv_shift_pct)) / 100)

        # 2. クロップとスケーリング
        x1, y1, x2, y2 = self._get_crop_box(lm, w, h)
        img_cropped = img[y1:y2, x1:x2]
        
        # パネルの高さ（1400px相当）に合わせてリサイズ
        target_ph = 1400 # 診断パネルの標準的な高さ
        scale = target_ph / img_cropped.shape[0]
        img_final = cv2.resize(img_cropped, (int(img_cropped.shape[1] * scale), target_ph), interpolation=cv2.INTER_CUBIC)
        fw, fh = img_final.shape[1], img_final.shape[0]

        # 3. 描画（リサイズ済みの最終画像に対して実施）
        draw_skeleton_zoom(img_final, lm, w, h, x1, y1, scale)
        draw_meas_line_zoom(img_final, lm[7],  lm[8],  sc_head,   w, h, x1, y1, scale)
        draw_meas_line_zoom(img_final, lm[23], lm[24], sc_pelvis, w, h, x1, y1, scale)
        draw_midline_zoom(img_final, lm, w, h, x1, y1, scale)

        # 筋肉緊張・重心 (Phase 23/24)
        tensions = estimate_muscle_tension(lm, 'front')
        draw_muscle_heatmap(img_final, lm, tensions, w, h, x1, y1, scale)
        draw_cog_indicator(img_final, lm, w, h, x1, y1, scale, 'front')

        # 4. ラベル描画（はみ出し防止付き）
        pil_photo = cv2pil(img_final); dp = ImageDraw.Draw(pil_photo); fl = get_font(20); fl_s = get_font(16)
        def lbl_zoom(lm_a, lm_b, text, sc):
            p1 = px_zoom(lm_a, w, h, x1, y1, scale); p2 = px_zoom(lm_b, w, h, x1, y1, scale)
            mx, my = (p1[0]+p2[0])//2, (p1[1]+p2[1])//2 - 25
            col = SCORE_RGB[sc]; bb = dp.textbbox((0,0), text, font=fl); tw = bb[2]-bb[0]; th = bb[3]-bb[1]
            # はみ出し防止 (X座標)
            draw_x = max(10, min(mx - tw//2, fw - tw - 10))
            dp.text((draw_x, my), text, font=fl, fill=col, stroke_width=2, stroke_fill=(10,12,20))

        lbl_zoom(lm[7],  lm[8],  f"頭部 {sc_head} {abs(head_a):.1f}°", sc_head)
        lbl_zoom(lm[11], lm[12], f"肩ライン {sc_shldr} {abs(shldr_a):.1f}°", sc_shldr)
        lbl_zoom(lm[23], lm[24], f"骨盤ライン {sc_pelvis} {abs(pelvis_a):.1f}°", sc_pelvis)

        # 正中線偏差ラベル
        lx = int(((lm[27].x + lm[28].x) / 2 * w - x1) * scale)
        for ml_x, ml_y, text in [
            (ear_mid_x, lm[7].y, f"{'←' if (ear_mid_x-foot_mid_x)<0 else ''}{abs(ear_shift_pct):.1f}%{'→' if (ear_mid_x-foot_mid_x)>=0 else ''}"),
            (shldr_mid_x, lm[11].y, f"{'←' if (shldr_mid_x-foot_mid_x)<0 else ''}{abs(shldr_shift_pct):.1f}%{'→' if (shldr_mid_x-foot_mid_x)>=0 else ''}"),
            (pelv_mid_x, lm[23].y, f"{'←' if (pelv_mid_x-foot_mid_x)<0 else ''}{abs(pelv_shift_pct):.1f}%{'→' if (pelv_mid_x-foot_mid_x)>=0 else ''}")
        ]:
            px, py = int((ml_x*w - x1)*scale), int((ml_y*h - y1)*scale)
            # はみ出し防止
            draw_x = max(10, min(px + 10, fw - 80))
            dp.text((draw_x, py - 5), text, font=fl_s, fill=MIDLINE_COL_RGB, stroke_width=1, stroke_fill=(10,12,20))

        img_final = pil2cv2(np.array(pil_photo))
        risk_msgs = calc_body_risks(sc_head, sc_shldr, sc_pelvis, ts_score, shldr_a, pelvis_a)
        score_items = [
            {"name": "頭部（耳の傾き）", "normal": 0.0, "measured": head_a, "diff": abs(head_a), "direction": direction(head_a), "score": sc_head},
            {"name": "肩ライン（傾き）", "normal": 0.0, "measured": shldr_a, "diff": abs(shldr_a), "direction": direction(shldr_a), "score": sc_shldr},
            {"name": "骨盤ライン（傾き）", "normal": 0.0, "measured": pelvis_a, "diff": abs(pelvis_a), "direction": direction(pelvis_a), "score": sc_pelvis},
            {"name": "頭部ズレ（正中線）", "normal": 0.0, "measured": ear_shift_pct, "diff": abs(ear_shift_pct), "direction": "右偏位" if ear_shift_pct > 0 else "左偏位", "score": ts_score},
            {"name": "肩部ズレ（正中線）", "normal": 0.0, "measured": shldr_shift_pct, "diff": abs(shldr_shift_pct), "direction": "右偏位" if shldr_shift_pct > 0 else "左偏位", "score": ts_score},
            {"name": "骨盤ズレ（正中線）", "normal": 0.0, "measured": pelv_shift_pct, "diff": abs(pelv_shift_pct), "direction": "右偏位" if pelv_shift_pct > 0 else "左偏位", "score": ts_score},
        ]
        panel = build_panel(score_items, risk_msgs, 560, fh)
        success = self._save_final_report(img_final, panel, output_path, "正面")
        
        # 数値データを辞書で返す
        data = {
            'head_angle': head_a,
            'shoulder_angle': shldr_a,
            'pelvis_angle': pelvis_a,
            'ear_shift_pct': ear_shift_pct,
            'shoulder_shift_pct': shldr_shift_pct,
            'pelvis_shift_pct': pelv_shift_pct
        }
        return success, data

    # ── 内部解析ロジック (側面) ──
    def _analyze_side(self, img, lm, w, h, output_path):
        # 1. 解析（数値計算のみ先に実施）
        mouth_x, ear_avg_x = (lm[9].x + lm[10].x) / 2, (lm[7].x + lm[8].x) / 2
        facing_right = mouth_x > ear_avg_x
        idx_ear = 7 if abs(lm[7].x - lm[0].x) > abs(lm[8].x - lm[0].x) else 8
        idx = [idx_ear, 12, 24, 26, 28] if idx_ear == 8 else [idx_ear, 11, 23, 25, 27]
        ear, shldr, hip, knee, ankle = [lm[i] for i in idx]
        ref_len = max(abs(shldr.y - hip.y) * h, 1)

        y_off = int(ref_len * 0.07); x_off = int(ref_len * 0.04) * (1 if not facing_right else -1)
        # 座標計算用の中間点（スケーリング前）
        ear_px_orig = (int(ear.x*w + x_off), int(ear.y*h + y_off))
        shldr_px_orig, hip_px_orig, ankle_px_orig = pxcoord(shldr,w,h), pxcoord(hip,w,h), pxcoord(ankle,w,h)

        fhp_pct = (ear_px_orig[0]/w - shldr.x) * w / ref_len * 100 * (1 if facing_right else -1)
        rs_pct  = (shldr.x - hip.x) * w / ref_len * 100 * (1 if facing_right else -1)
        pel_a   = math.degrees(math.atan2((knee.x-hip.x)*w, (knee.y-hip.y)*h)) * (1 if facing_right else -1)
        trunk_pct = (ear_px_orig[0]/w - ankle.x) * w / ref_len * 100 * (1 if facing_right else -1)
        fhp_sc, rs_sc, pel_sc, trk_sc = [_get_side_score(k, abs(v)/100 if "pct" in k else abs(v)) for k, v in zip(["FHP","ラウンドショルダー","骨盤前後傾","体幹ライン"], [fhp_pct, rs_pct, pel_a, trunk_pct])]

        # 2. クロップとスケーリング
        x1, y1, x2, y2 = self._get_crop_box(lm, w, h)
        img_cropped = img[y1:y2, x1:x2]
        target_ph = 1400
        scale = target_ph / img_cropped.shape[0]
        img_final = cv2.resize(img_cropped, (int(img_cropped.shape[1] * scale), target_ph), interpolation=cv2.INTER_CUBIC)
        fw, fh = img_final.shape[1], img_final.shape[0]

        # 3. 描画
        draw_skeleton_zoom(img_final, lm, w, h, x1, y1, scale)
        # 重心線
        ankle_px = px_zoom(ankle, w, h, x1, y1, scale)
        ear_px = (int((ear_px_orig[0]-x1)*scale), int((ear_px_orig[1]-y1)*scale))
        shldr_px = px_zoom(shldr, w, h, x1, y1, scale)
        hip_px = px_zoom(hip, w, h, x1, y1, scale)
        
        ankle_top_y = max(ear_px[1] - 50, 20)
        for yy in range(ankle_top_y, ankle_px[1], 25):
            cv2.line(img_final, (ankle_px[0], yy), (ankle_px[0], min(yy+12, ankle_px[1])), MIDLINE_COL_BGR, 2, cv2.LINE_AA)
        
        pts = [ear_px, shldr_px, hip_px, ankle_px]
        for i in range(len(pts)-1): cv2.line(img_final, pts[i], pts[i+1], (200,200,50), 2, cv2.LINE_AA)
        for pt in pts: cv2.circle(img_final, pt, 7, (200,200,50), -1)

        # 筋肉緊張・重心 (Phase 23/24)
        tensions = estimate_muscle_tension(lm, 'side')
        draw_muscle_heatmap(img_final, lm, tensions, w, h, x1, y1, scale)
        draw_cog_indicator(img_final, lm, w, h, x1, y1, scale, 'side')

        # 水平偏差矢印
        def draw_h_diff_zoom(p_a, p_b, sc):
            col = SCORE_BGR[sc]; my = (p_a[1]+p_b[1])//2
            cv2.arrowedLine(img_final, (p_b[0],my), (p_a[0],my), col, 2, cv2.LINE_AA, tipLength=0.2)
            cv2.circle(img_final, p_a, 7, col, -1); cv2.circle(img_final, p_b, 7, col, -1)
        draw_h_diff_zoom(ear_px, shldr_px, fhp_sc)

        # 4. ラベル描画（はみ出し防止）
        pil_p = cv2pil(img_final); dp = ImageDraw.Draw(pil_p); fl = get_font(20)
        def s_lbl_zoom(pt, txt, sc, dy=-25):
            col = SCORE_RGB[sc]; tx, ty = pt[0]+12, pt[1]+dy
            bb = dp.textbbox((0,0),txt,font=fl); tw = bb[2]-bb[0]
            # はみ出し防止 (左右)
            draw_x = tx if tx + tw < fw - 10 else pt[0] - tw - 12
            draw_x = max(10, min(draw_x, fw - tw - 10))
            dp.text((draw_x, ty), txt, font=fl, fill=col, stroke_width=2, stroke_fill=(10,12,20))

        s_lbl_zoom(ear_px, f"耳垂 FHP:{abs(fhp_pct):.0f}% {fhp_sc}", fhp_sc)
        s_lbl_zoom(shldr_px, f"肩 RS:{abs(rs_pct):.0f}% {rs_sc}", rs_sc)
        s_lbl_zoom(hip_px, f"股 骨盤:{abs(pel_a):.1f}° {pel_sc}", pel_sc, dy=5)
        s_lbl_zoom(ankle_px, f"足 体幹:{abs(trunk_pct):.0f}% {trk_sc}", trk_sc, dy=5)
        
        img_final = pil2cv2(np.array(pil_p))
        risk_msgs = calc_side_risks(fhp_sc, rs_sc, trk_sc, pel_sc, abs(fhp_pct), abs(rs_pct))
        score_items = [
            {"name":"前方頭位(FHP)","ideal":"0%","measured":f"{fhp_pct:+.1f}%","diff":f"{abs(fhp_pct):.1f}%","score":fhp_sc},
            {"name":"ラウンドショルダー","ideal":"0%","measured":f"{rs_pct:+.1f}%","diff":f"{abs(rs_pct):.1f}%","score":rs_sc},
            {"name":"骨盤前後傾","ideal":"0°","measured":f"{pel_a:+.1f}°","diff":f"{abs(pel_a):.1f}°","score":pel_sc},
            {"name":"体幹ライン領域","ideal":"0%","measured":f"{trunk_pct:+.1f}%","diff":f"{abs(trunk_pct):.1f}%","score":trk_sc},
        ]
        panel = build_side_panel(score_items, risk_msgs, 560, fh)
        success = self._save_final_report(img_final, panel, output_path, f"側面：{'右向き' if facing_right else '左向き'}")
        
        # 数値データを辞書で返す
        data = {
            'fhp_pct': fhp_pct,
            'rs_pct': rs_pct,
            'pelvis_angle': pel_a,
            'trunk_pct': trunk_pct
        }
        return success, data

    def _get_crop_box(self, lm, w, h):
        """人物の存在範囲から切り抜き範囲(x1, y1, x2, y2)を計算する"""
        pts = [0, 7, 8, 11, 12, 23, 24, 25, 26, 27, 28]
        xs = [lm[i].x for i in pts if lm[i].visibility > 0.3]
        ys = [lm[i].y for i in pts if lm[i].visibility > 0.3]
        if not xs or not ys: return 0, 0, w, h
        min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
        person_h = max_y - min_y
        m_top, m_bot, m_side = person_h * 0.15, person_h * 0.1, person_h * 0.1
        return (max(0, int((min_x - m_side) * w)), max(0, int((min_y - m_top) * h)),
                min(w, int((max_x + m_side) * w)), min(h, int((max_y + m_bot) * h)))

    def _save_final_report(self, img, panel, output_path, title_suffix):
        # 描画済みのimgとpanelを連結して保存
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
def draw_text(draw, pos, text, font, color): draw.text(pos, text, font=font, fill=color, stroke_width=1, stroke_fill=color)
def draw_text_center(draw, cx, y, text, font, color):
    bb = draw.textbbox((0, 0), text, font=font); tw = bb[2] - bb[0]
    draw.text((cx - tw // 2, y), text, font=font, fill=color, stroke_width=1, stroke_fill=color)

CONNECTIONS = [(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]
def px_zoom(lm, w, h, x1, y1, scale):
    """ズーム・クロップ後の座標に変換する"""
    return (int((lm.x * w - x1) * scale), int((lm.y * h - y1) * scale))

def draw_skeleton_zoom(img, lm, w, h, x1, y1, scale):
    for s, e in CONNECTIONS:
        if lm[s].visibility > 0.3 and lm[e].visibility > 0.3:
            p1 = px_zoom(lm[s], w, h, x1, y1, scale)
            p2 = px_zoom(lm[e], w, h, x1, y1, scale)
            cv2.line(img, p1, p2, (150,150,150), 1, cv2.LINE_AA)
    for l in lm:
        cv2.circle(img, px_zoom(l, w, h, x1, y1, scale), 3, (80,230,120), -1)

def draw_meas_line_zoom(img, lm1, lm2, sc, w, h, x1, y1, scale):
    c = SCORE_BGR[sc]
    p1 = px_zoom(lm1, w, h, x1, y1, scale)
    p2 = px_zoom(lm2, w, h, x1, y1, scale)
    # 線を細く（2px）に変更して精密感を出す
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
    
    pts_lm = [lm[0], lm[11], lm[23], lm[27]] # 適当な代表点
    # 現実に即してランドマーク中点を計算
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

# ─── 高度な解析演出 (重心・筋肉) ───────────────────────────────────────────
def estimate_muscle_tension(landmarks, view):
    """姿勢ランドマークから筋肉の緊張度(0.0-1.0)を部位別に推定"""
    tensions = {}
    lm = landmarks
    if view == 'front':
        # 正面視：左右差に基づく推定
        shldr_a = abs(calc_angle(lm[11], lm[12], 100, 100))
        pelvis_a = abs(calc_angle(lm[23], lm[24], 100, 100))
        tensions['trapezius_l'] = min(shldr_a / 5.0, 1.0) if lm[11].y > lm[12].y else 0.1
        tensions['trapezius_r'] = min(shldr_a / 5.0, 1.0) if lm[12].y > lm[11].y else 0.1
        tensions['erector_spinae_l'] = min(pelvis_a / 4.0, 1.0) if lm[23].y > lm[24].y else 0.2
        tensions['erector_spinae_r'] = min(pelvis_a / 4.0, 1.0) if lm[24].y > lm[23].y else 0.2
    else:
        # 側面視：前後ズレに基づく推定
        # 首の筋肉 (Splenius/SCM)
        fhp = abs(lm[0].x - lm[11].x) # 簡易的なFHP
        tensions['neck_extensor'] = min(fhp * 5.0, 1.0)
        # 腰 (Lumbar)
        tensions['lumbar_extensor'] = min(abs(lm[11].x - lm[23].x) * 3.0, 0.8)
        # 前もも/裏もも
        tensions['quads'] = min(abs(lm[23].x - lm[25].x) * 4.0, 0.7)
    return tensions

def draw_muscle_heatmap(img, landmarks, tensions, w, h, x1, y1, scale):
    """筋肉の緊張箇所を半透明のヒートマップでオーバーレイ"""
    overlay = img.copy()
    lm = landmarks
    font = get_font(12)
    
    def draw_tension_blob(point_lm, tension, color=(0, 0, 255), label=""):
        if tension < 0.3: return
        p = px_zoom(point_lm, w, h, x1, y1, scale)
        radius = int(30 * tension * scale)
        alpha = int(150 * tension)
        # ぼかしの効いた円を描画
        cv2.circle(overlay, p, radius, color, -1)
        if label:
            cv2.putText(img, label, (p[0]+15, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    # 部位別の緊張度を可視化
    for part, t in tensions.items():
        if part.startswith('trapezius'):
            idx = 11 if '_l' in part else 12
            draw_tension_blob(lm[idx], t, (0, 100, 255), "Trapezius")
        elif part == 'neck_extensor':
            # 首の付け根付近
            class MidPoint:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
                    self.visibility = 1.0
            mn = MidPoint((lm[7].x+lm[8].x)/2, (lm[7].y+lm[11].y)/2)
            draw_tension_blob(mn, t, (0, 120, 255), "Neck Stress")
        elif part == 'erector_spinae':
            idx = 23 if '_l' in part else 24
            draw_tension_blob(lm[idx], t, (0, 80, 220), "Back Stress")

    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

def draw_cog_indicator(img, lm, w, h, x1, y1, scale, view):
    """支持基底面に対する重心(COG)位置と左右荷重バランスを表示"""
    # 両足の中点を計算
    ankle_l, ankle_r = lm[27], lm[28]
    base_mid_x = (ankle_l.x + ankle_r.x) / 2
    
    # 上半身の簡易重心（鼻、肩、腰の中点）
    upper_body_x = (lm[0].x + lm[11].x + lm[12].x + lm[23].x + lm[24].x) / 5
    
    # ズレを計算 (-1.0 to 1.0)
    width = max(abs(ankle_l.x - ankle_r.x), 0.05)
    shift = (upper_body_x - base_mid_x) / width
    
    # 足元座標
    p_base = midpoint(px_zoom(ankle_l, w, h, x1, y1, scale), px_zoom(ankle_r, w, h, x1, y1, scale))
    py = p_base[1] + 40
    
    if view == 'front':
        # 荷重バランスインジケータ（正面）
        bar_w = 120
        cv2.rectangle(img, (p_base[0]-bar_w//2, py), (p_base[0]+bar_w//2, py+8), (60,60,60), -1)
        # 現在の重心点
        cog_x = int(p_base[0] + (shift * bar_w / 2))
        cog_x = max(p_base[0]-bar_w//2, min(p_base[0]+bar_w//2, cog_x))
        cv2.circle(img, (cog_x, py+4), 6, (0, 255, 255), -1)
        
        # パーセント表示 (見認性向上のために太字・赤色・縁取りを追加)
        left_p = 50 - (shift * 50)
        right_p = 100 - left_p
        
        def put_text_bold(text, pos, color, scale=0.6, thickness=2):
            # 縁取り（黒）
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (10, 10, 10), thickness + 2, cv2.LINE_AA)
            # 本体
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

        # 荷重が大きい方を強調（赤系、小さい方は白系）
        col_l = (80, 80, 255) if left_p > 55 else (240, 240, 240)
        col_r = (80, 80, 255) if right_p > 55 else (240, 240, 240)
        
        put_text_bold(f"L:{left_p:.0f}%", (p_base[0]-bar_w//2-75, py+15), col_l)
        put_text_bold(f"R:{right_p:.0f}%", (p_base[0]+bar_w//2+15, py+15), col_r)
        
        # 中央ラベル
        put_text_bold("LOAD BALANCE", (p_base[0]-65, py+38), (80, 220, 240), 0.45, 1)
    else:
        # 重心フラグ（側面）
        direction = "FORWARD" if shift > 0.1 else "BACKWARD" if shift < -0.1 else "IDEAL"
        color = (80, 80, 255) if direction != "IDEAL" else (80, 220, 140)
        
        def put_text_bold_side(text, pos, color, scale=0.6, thickness=2):
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (10, 10, 10), thickness + 2, cv2.LINE_AA)
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
            
        put_text_bold_side(f"COG: {direction}", (p_base[0]-60, py+45), color)

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

def build_comparison_panel(it1, it2, risks, pw, ih, mode):
    p = Image.new("RGB", (pw, ih), PANEL_BG); dr = ImageDraw.Draw(p); fT, fH, fB, fS, fXS, fXXS = get_font(28), get_font(22), get_font(19), get_font(16), get_font(14), get_font(13)
    dr.rectangle([(0,0),(pw,50)], fill=(30,40,70)); draw_text_center(dr, pw//2, 8, f"🩺 姿勢変化・改善指標（{mode}）", fT, WHITE)
    y = 65; draw_text(dr, (18,y), "▌ 姿勢データの変化", fH, WHITE); y += 38
    
    for i, (b, a) in enumerate(zip(it1, it2)):
        dr.rectangle([(10,y),(pw-10,y+78)], fill=(34,40,68), outline=LINE_COL); draw_text(dr, (18,y+8), b["n"], fB, WHITE)
        # スコア変化
        def draw_sc(xx, yy, sc, label):
            col = SCORE_RGB[sc]; dr.ellipse([(xx,yy),(xx+18,yy+18)], fill=col); draw_text(dr, (xx+3,yy+1), sc, fXXS, (10,10,10)); draw_text(dr, (xx+25,yy+2), label, fXXS, GRAY)
        draw_sc(pw-140, y+8, b["s"], "前"); draw_sc(pw-60, y+8, a["s"], "後")
        
        # 数値比較
        unit = "°" if "傾き" in b["n"] or "傾" in b["n"] else "%"
        draw_text(dr, (18,y+42), f"Before: {abs(b['v']):.1f}{unit}", fS, GRAY); draw_text(dr, (160,y+42), f"After: {abs(a['v']):.1f}{unit}", fS, YELLOW)
        
        # 改善度（矢印）
        diff = abs(b["v"]) - abs(a["v"]); col_diff = (80,255,150) if diff > 0.1 else (255,100,80) if diff < -0.1 else GRAY
        txt_diff = f"改善: {diff:+.1f}{unit}" if diff > 0.1 else f"変化: {diff:+.1f}{unit}"
        draw_text(dr, (pw-140, y+42), txt_diff, fS, col_diff)
        y += 88

    y += 10; dr.line([(10,y),(pw-10,y)], fill=(70,78,120), width=2); y += 15; draw_text(dr, (18,y), "▌ 専門スタッフの視点", fH, WHITE); y += 22; draw_text(dr, (18,y), "最新の解析データに基づいたリスク評価です", fXXS, GRAY); y += 20
    for pt, sc, msg in risks:
        col = SCORE_RGB[sc]; dr.line([(10,y),(pw-10,y)], fill=LINE_COL); y += 8; draw_text(dr, (18,y), f"【{pt}】", fS, WHITE); bx = 18+70; dr.ellipse([(bx,y-2),(bx+20,y+18)], fill=col); draw_text(dr, (bx+3,y-1), sc, fXXS, (10,10,10))
        rem = msg; wrp = []
        while len(rem) > MAX_CHARS: wrp.append(rem[:MAX_CHARS]); rem = rem[MAX_CHARS:]
        if rem: wrp.append(rem)
        for i, t in enumerate(wrp): draw_text(dr, (18+105, y+i*15), t, fXXS, col)
        y += 40
    y += 10; dr.line([(10,y),(pw-10,y)], fill=(70,78,120), width=2); y += 15
    draw_text(dr, (pw//2-120, y), "整体院 導 ｜ 技術提携：AI 姿勢解析エンジン", fXXS, GRAY)
    return pil2cv2(np.array(p))

if __name__ == "__main__":
    TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(TEST_DIR, "pose_landmarker.task")
    engine = PoseAnalyzer(MODEL_PATH)
    targets = glob.glob(os.path.join(TEST_DIR, "*.jpg"))
    for t in targets:
        if "annotated" in t: continue
        engine.analyze(t, os.path.join(TEST_DIR, f"annotated_v9_{os.path.basename(t)}"))
