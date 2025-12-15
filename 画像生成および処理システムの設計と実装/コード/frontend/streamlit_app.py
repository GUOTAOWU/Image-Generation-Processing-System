# streamlit run frontend/streamlit_app.py

import io
import requests
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import os

# バックエンドのURL設定（環境変数から取得、デフォルトはローカルホスト）
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# ページ設定
st.set_page_config(page_title="画像生成・処理システム", layout="centered")
st.title("画像生成・処理システム")


# 互換性のためのリロード関数
# Streamlitのバージョンによってリロード方法が異なるため、複数を試行します
def safe_rerun():
    if hasattr(st, "experimental_rerun"):
        try:
            st.experimental_rerun()
            return
        except Exception:
            pass
    if hasattr(st, "rerun"):
        try:
            st.rerun()
            return
        except Exception:
            pass
    # フォールバックとしてセッション状態を使用
    st.session_state["_need_refresh"] = True


# セッション状態の初期化
if "token" not in st.session_state:
    st.session_state["token"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None


# クエリパラメータからページ情報を取得
def _get_page_from_query():
    qp = st.query_params
    val = qp.get("page", None)
    if val is None:
        return "login"
    if isinstance(val, list):
        return val[0] if len(val) > 0 else "login"
    return str(val)


# ページ遷移用関数
def set_page(page_name: str):
    st.session_state["page"] = page_name


# 初期ページの設定
if "page" not in st.session_state:
    st.session_state["page"] = _get_page_from_query() or "login"


# ---------- ヘルパー関数: API呼び出し ----------

def register_user(username: str, password: str):
    """ユーザー登録APIを呼び出します"""
    payload = {"username": username, "password": password}
    resp = requests.post(f"{BACKEND_URL}/register", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def login_user(username: str, password: str):
    """ログインAPIを呼び出し、トークンを取得します"""
    payload = {"username": username, "password": password}
    resp = requests.post(f"{BACKEND_URL}/login", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def upload_and_ocr(file_bytes, filename, mime_type, token):
    """OCRエンドポイントへ画像をアップロードします"""
    files = {"file": (filename, file_bytes, mime_type)}
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.post(f"{BACKEND_URL}/ocr", files=files, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.json()


def upload_and_remove_bg(file_bytes, filename, mime_type, token):
    """背景除去エンドポイントへ画像をアップロードします"""
    files = {"file": (filename, file_bytes, mime_type)}
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.post(f"{BACKEND_URL}/remove_bg", files=files, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.content  # PNGのバイトデータを返す


def upload_and_crop(file_bytes, filename, mime_type, token, x1, y1, x2, y2):
    """画像切り抜き（クロップ）エンドポイントへリクエストを送ります"""
    files = {"file": (filename, file_bytes, mime_type)}
    data = {"x1": str(x1), "y1": str(y1), "x2": str(x2), "y2": str(y2)}
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.post(f"{BACKEND_URL}/crop", files=files, data=data, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.content  # PNGのバイトデータを返す


def upload_and_resize(file_bytes, filename, mime_type, token, mode, scale=None, width=None, height=None,
                      out_format="png", quality=90):
    """画像リサイズエンドポイントへリクエストを送ります"""
    files = {"file": (filename, file_bytes, mime_type)}
    data = {"mode": mode, "out_format": out_format, "quality": str(int(quality))}
    if scale is not None:
        data["scale"] = str(float(scale))
    if width is not None:
        data["width"] = str(int(width))
    if height is not None:
        data["height"] = str(int(height))
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.post(f"{BACKEND_URL}/resize", files=files, data=data, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.content  # 画像のバイトデータを返す


# 文生図（テキストから画像生成）呼び出し
def synth_image(prompt, token, n=1, size="1328*1328", prompt_extend=True, watermark=True, api_key=None):
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "prompt": prompt,
        "n": int(n),
        "size": size,
        "prompt_extend": bool(prompt_extend),
        "watermark": bool(watermark),
    }
    if api_key:
        payload["api_key"] = api_key
    resp = requests.post(f"{BACKEND_URL}/synth_image", json=payload, headers=headers, timeout=300)
    resp.raise_for_status()
    ct = resp.headers.get("Content-Type", "")
    return resp.content, ct


# ---------- サイドバー ----------
def show_sidebar():
    st.sidebar.title("ナビゲーション")
    if st.session_state.get("token"):
        st.sidebar.write(f"ログイン中: **{st.session_state['username']}**")
        if st.sidebar.button("ログアウト"):
            st.session_state["token"] = None
            st.session_state["username"] = None
            set_page("login")
            safe_rerun()
            return

        st.sidebar.markdown("---")
        label_map = {
            "main": "メインメニュー",
            "ocr": "文字抽出 (OCR)",
            "removebg": "背景除去",
            "crop": "切り抜き",
            "resize": "解像度変更",
            "synth": "画像生成",
            "about": "このアプリについて"
        }
        options = list(label_map.values())
        current_label = label_map.get(st.session_state.get("page", "main"), "メインメニュー")
        choice = st.sidebar.radio("ページ", options, index=options.index(current_label), key="sidebar_page")
        label_to_page = {v: k for k, v in label_map.items()}
        selected_page = label_to_page.get(choice, "main")
        if selected_page != st.session_state.get("page"):
            st.session_state["page"] = selected_page
            safe_rerun()
    else:
        st.sidebar.write("未ログイン")
        options = ["ログイン / 登録", "このアプリについて"]
        choice = st.sidebar.radio("ページ", options, index=0, key="sidebar_anon")
        if choice == "ログイン / 登録":
            if st.session_state.get("page") not in ("login", "register"):
                set_page("login")
                safe_rerun()
        else:
            if st.session_state.get("page") != "about":
                set_page("about")
                safe_rerun()


show_sidebar()


# ---------- ページ実装 ----------

def page_login():
    st.header("ログイン")
    col1, col2 = st.columns([2, 1])
    with col1:
        username = st.text_input("ユーザー名", key="login_username")
        password = st.text_input("パスワード", type="password", key="login_password")
        if st.button("ログイン"):
            try:
                with st.spinner("ログイン中..."):
                    data = login_user(username, password)
                    st.session_state["token"] = data["access_token"]
                    st.session_state["username"] = username
                    st.success("ログイン成功、メインページへ移動します")
                    set_page("main")
                    safe_rerun()
            except Exception as e:
                st.error(f"ログイン失敗: {e}")

    with col2:
        st.write("アカウントをお持ちでないですか？")
        if st.button("新規登録へ"):
            set_page("register")
            safe_rerun()


def page_register():
    st.header("新規アカウント登録")
    username = st.text_input("ユーザー名（登録）", key="reg_username")
    password = st.text_input("パスワード", type="password", key="reg_password")
    password2 = st.text_input("パスワード（確認用）", type="password", key="reg_password2")
    if st.button("登録してログイン"):
        if not username or not password:
            st.error("ユーザー名とパスワードは必須です")
        elif password != password2:
            st.error("パスワードが一致しません")
        else:
            try:
                with st.spinner("登録中..."):
                    data = register_user(username, password)
                    st.session_state["token"] = data["access_token"]
                    st.session_state["username"] = username
                    st.success("登録成功、ログインしました。メインページへ移動します")
                    set_page("main")
                    safe_rerun()
            except Exception as e:
                st.error(f"登録失敗: {e}")

    if st.button("ログイン画面へ戻る"):
        set_page("login")
        safe_rerun()


def page_main():
    st.header("メインメニュー")
    st.write("以下の機能を選択してください：")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("文字抽出（OCR）", key="go_ocr", help="OCRページへ移動"):
            set_page("ocr")
            safe_rerun()
    with c2:
        if st.button("背景除去（Matting）", key="go_removebg", help="背景除去ページへ移動"):
            set_page("removebg")
            safe_rerun()
    c3, c4 = st.columns(2)
    with c3:
        if st.button("画像切り抜き（Crop）", key="go_crop", help="切り抜きページへ移動"):
            set_page("crop")
            safe_rerun()
    with c4:
        if st.button("解像度変更", key="go_resize", help="解像度変更ページへ移動"):
            set_page("resize")
            safe_rerun()
    c5, c6 = st.columns(2)
    with c5:
        if st.button("画像生成（文生図）", key="go_synth", help="画像生成ページへ移動"):
            set_page("synth")
            safe_rerun()
    st.write("---")
    st.write("その他：")
    col3, col4 = st.columns(2)
    with col3:
        if st.button("このアプリについて"):
            set_page("about")
            safe_rerun()


def page_ocr():
    st.header("文字抽出（OCR）")
    uploaded_file = st.file_uploader("OCR用の画像を選択してください（png/jpg/jpeg/bmp）",
                                     type=["png", "jpg", "jpeg", "bmp"], key="ocr_uploader")
    col1, col2 = st.columns(2)
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"画像を開けません: {e}")
            st.stop()

        col1.image(image, caption="元画像", use_column_width=True)

        if st.button("アップロードして識別"):
            token = st.session_state.get("token")
            if not token:
                st.error("ログイン情報が見つかりません。再ログインしてください")
                set_page("login")
                safe_rerun()
                st.stop()
            try:
                with st.spinner("識別中..."):
                    resp = upload_and_ocr(uploaded_file.getvalue(), uploaded_file.name, uploaded_file.type, token)
            except Exception as e:
                st.error(f"識別リクエスト失敗: {e}")
                st.stop()

            # 結果の表示
            raw_text = resp.get("raw_text", "")
            st.subheader("識別結果（テキスト原文）")
            st.text_area("検出されたテキスト", raw_text, height=200)

            # 画像上にバウンディングボックスを描画
            lines = resp.get("lines", [])
            draw_img = image.copy()
            draw = ImageDraw.Draw(draw_img)
            try:
                font = ImageFont.truetype("arial.ttf", size=18)
            except Exception:
                font = ImageFont.load_default()

            w, h = draw_img.size
            for item in lines:
                bbox = item.get("bbox", [])
                text = item.get("text", "")
                if not bbox:
                    continue
                # 相対座標から絶対座標へ変換
                pts = [(int(x * w), int(y * h)) for x, y in bbox]
                if len(pts) >= 3:
                    draw.line(pts + [pts[0]], width=3, fill=(255, 0, 0))

                # テキストラベルの描画
                tx, ty = pts[0]
                text_bbox = draw.textbbox((tx, ty), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                padding = 3
                rect = [tx, ty - text_height - padding * 2, tx + text_width + padding * 2, ty]
                draw.rectangle(rect, fill=(255, 0, 0))
                draw.text((tx + padding, ty - text_height - padding), text, fill=(255, 255, 255), font=font)

            col2.image(draw_img, caption="識別エリア付き画像", use_column_width=True)

    if st.button("メインメニューに戻る"):
        set_page("main")
        safe_rerun()


def page_removebg():
    st.header("背景除去（Matting）")
    uploaded_file = st.file_uploader("画像を選択してください（png/jpg/jpeg/bmp）", type=["png", "jpg", "jpeg", "bmp"],
                                     key="rm_bg_uploader")
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"画像を開けません: {e}")
            st.stop()
        st.image(image, caption="元画像", use_column_width=True)
        if st.button("背景を除去してダウンロード"):
            token = st.session_state.get("token")
            if not token:
                st.error("ログイン情報が見つかりません。再ログインしてください")
                set_page("login")
                safe_rerun()
                st.stop()
            try:
                with st.spinner("処理中...（初回は時間がかかる場合があります）"):
                    png_bytes = upload_and_remove_bg(uploaded_file.getvalue(), uploaded_file.name, uploaded_file.type,
                                                     token)
            except Exception as e:
                st.error(f"リクエスト失敗: {e}")
                st.stop()

            st.success("背景除去完了（PNG形式）")
            st.image(png_bytes, caption="処理結果（透明背景）", use_column_width=True)
            st.download_button("PNGをダウンロード（透明背景）", data=png_bytes, file_name="result.png", mime="image/png")

    if st.button("メインメニューに戻る", key="rm_back"):
        set_page("main")
        safe_rerun()


def page_crop():
    st.header("画像切り抜き（Crop）")
    st.write("画像をアップロード -> スライダーで範囲選択（％） -> プレビュー -> 切り抜いてダウンロード（バックエンド処理）")
    uploaded_file = st.file_uploader("画像を選択してください（png/jpg/jpeg/bmp）", type=["png", "jpg", "jpeg", "bmp"],
                                     key="crop_uploader")
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGBA")
        except Exception as e:
            st.error(f"画像を開けません: {e}")
            st.stop()

        w, h = image.size
        st.markdown(f"**元画像サイズ：** {w} × {h}（ピクセル）")
        col_preview, col_controls = st.columns([2, 1])

        with col_controls:
            st.write("切り抜き範囲の選択（％）")
            left_pct = st.slider("左 (L %)", min_value=0, max_value=99, value=5, step=1)
            top_pct = st.slider("上 (T %)", min_value=0, max_value=99, value=5, step=1)
            right_pct = st.slider("右 (R %)", min_value=1, max_value=100, value=95, step=1)
            bottom_pct = st.slider("下 (B %)", min_value=1, max_value=100, value=95, step=1)

            # プレビューボタン
            if st.button("プレビュー"):
                left = int(left_pct / 100.0 * w)
                top = int(top_pct / 100.0 * h)
                right = int(right_pct / 100.0 * w)
                bottom = int(bottom_pct / 100.0 * h)
                if right <= left or bottom <= top:
                    st.error("無効な切り抜き領域です。左/右または上/下の値を確認してください。")
                else:
                    cropped_preview = image.crop((left, top, right, bottom))
                    col_preview.image(cropped_preview, caption=f"プレビュー：{right - left}×{bottom - top}",
                                      use_column_width=True)

            # バックエンド呼び出し
            if st.button("切り抜いてダウンロード"):
                token = st.session_state.get("token")
                if not token:
                    st.error("ログイン情報が見つかりません。再ログインしてください")
                    set_page("login")
                    safe_rerun()
                    st.stop()
                x1 = left_pct / 100.0
                y1 = top_pct / 100.0
                x2 = right_pct / 100.0
                y2 = bottom_pct / 100.0
                try:
                    with st.spinner("切り抜き処理中..."):
                        png_bytes = upload_and_crop(uploaded_file.getvalue(), uploaded_file.name, uploaded_file.type,
                                                    token, x1, y1, x2, y2)
                except Exception as e:
                    st.error(f"リクエスト失敗: {e}")
                    st.stop()

                st.success("完了（PNG形式）")
                st.image(png_bytes, caption="切り抜き結果（バックエンド処理）", use_column_width=True)
                st.download_button("ダウンロード", data=png_bytes, file_name="cropped.png", mime="image/png")

        with col_preview:
            st.image(image, caption="元画像", use_column_width=True)

    if st.button("メインメニューに戻る", key="crop_back"):
        set_page("main")
        safe_rerun()


def page_resize():
    st.header("解像度・画質変更（Resize / Quality）")
    st.write("画像をアップロード -> 縮小方式を選択（倍率/ピクセル）-> プレビュー -> バックエンドで処理してダウンロード")
    uploaded_file = st.file_uploader("調整用の画像を選択してください（png/jpg/jpeg/bmp）",
                                     type=["png", "jpg", "jpeg", "bmp"], key="resize_uploader")
    if not uploaded_file:
        st.info("まずは画像をアップロードしてください")
    else:
        try:
            orig_img = Image.open(uploaded_file).convert("RGBA")
        except Exception as e:
            st.error(f"画像を開けません: {e}")
            st.stop()

        w, h = orig_img.size
        st.markdown(f"**元画像サイズ：** {w} × {h}（ピクセル）")

        col_preview, col_controls = st.columns([2, 1])
        with col_controls:
            st.write("リサイズモード")
            mode = st.radio("モード選択", options=["倍率指定（Scale）", "ピクセル指定（Dims）"], index=0, key="resize_mode")
            out_format = st.selectbox("出力フォーマット", options=["png", "jpeg"], index=0, key="resize_format")

            # パラメータ入力
            if mode == "倍率指定（Scale）":
                scale_pct = st.slider("倍率（%）", min_value=10, max_value=500, value=100, step=1, key="resize_scale_pct")
                scale = scale_pct / 100.0
                width_input = None
                height_input = None
                st.write(f"予想サイズ：{int(round(w * scale))} × {int(round(h * scale))}")
            else:
                width_input = st.number_input("目標幅（px）", min_value=1, step=1, value=w, key="resize_width")
                height_input = st.number_input("目標高さ（px）", min_value=1, step=1, value=h, key="resize_height")
                scale = None

            if out_format == "jpeg":
                quality = st.slider("JPEG 画質（Quality）", min_value=10, max_value=95, value=90, step=1,
                                    key="resize_quality")
            else:
                quality = 90

            # ローカルプレビュー
            if st.button("プレビュー（ローカル）"):
                try:
                    if mode == "倍率指定（Scale）":
                        new_w = max(1, int(round(w * scale)))
                        new_h = max(1, int(round(h * scale)))
                    else:
                        wi = int(width_input) if width_input else None
                        he = int(height_input) if height_input else None
                        if wi is None and he is None:
                            st.error("幅または高さを指定してください")
                            st.stop()
                        if wi is None:
                            scale_h = he / float(h)
                            new_h = max(1, int(he))
                            new_w = max(1, int(round(w * scale_h)))
                        elif he is None:
                            scale_w = wi / float(w)
                            new_w = max(1, int(wi))
                            new_h = max(1, int(round(h * scale_w)))
                        else:
                            new_w = max(1, int(wi))
                            new_h = max(1, int(he))

                    preview_img = orig_img.resize((new_w, new_h), Image.LANCZOS)
                    col_preview.image(preview_img, caption=f"プレビュー：{new_w}×{new_h}", use_column_width=True)
                except Exception as e:
                    st.error(f"プレビュー失敗: {e}")

            # バックエンド実行
            if st.button("処理してダウンロード（バックエンド）"):
                token = st.session_state.get("token")
                if not token:
                    st.error("ログイン情報が見つかりません。再ログインしてください")
                    set_page("login")
                    safe_rerun()
                    st.stop()

                try:
                    if mode == "倍率指定（Scale）":
                        s = scale
                        resp_bytes = upload_and_resize(uploaded_file.getvalue(), uploaded_file.name, uploaded_file.type,
                                                       token, mode="scale", scale=s, out_format=out_format,
                                                       quality=quality)
                    else:
                        wi = int(width_input) if width_input else None
                        he = int(height_input) if height_input else None
                        resp_bytes = upload_and_resize(uploaded_file.getvalue(), uploaded_file.name, uploaded_file.type,
                                                       token, mode="dims", width=wi, height=he, out_format=out_format,
                                                       quality=quality)
                except Exception as e:
                    st.error(f"リクエスト失敗: {e}")
                    st.stop()

                st.success("処理完了")
                st.image(resp_bytes, caption="処理結果", use_column_width=True)
                ext = "jpg" if out_format == "jpeg" else "png"
                st.download_button(f"{ext.upper()}ファイルをダウンロード", data=resp_bytes, file_name=f"resized.{ext}",
                                   mime=f"image/{ext}")

        with col_preview:
            st.image(orig_img, caption="元画像", use_column_width=True)

    if st.button("メインメニューに戻る", key="resize_back"):
        set_page("main")
        safe_rerun()


# ---------- 画像生成（文生図） ----------
def page_synth():
    st.header("画像生成（Dashscope / qwen-image）")
    st.write("プロンプトを入力し、パラメータを設定して画像を生成します。")
    prompt = st.text_area("プロンプト（日本語/英語どちらでも可）", height=220, key="synth_prompt")
    col1, col2 = st.columns([2, 1])
    with col2:
        n = st.number_input("生成枚数 (n)", min_value=1, max_value=4, value=1, step=1)
        size = st.selectbox("画像サイズ", options=["512*512", "768*768", "1024*1024", "1328*1328"], index=3)
        prompt_extend = st.checkbox("プロンプト自動拡張 (prompt_extend)", value=True)
        watermark = st.checkbox("透かしを入れる (watermark)", value=True)
        api_key_input = st.text_input("APIキーを入力 (任意)", value="", type="password",
                                      help="サーバー側の環境変数を使用する場合は空欄のままにしてください")
        if api_key_input.strip() == "":
            api_key_input = None

    if st.button("画像を生成（バックエンド）"):
        token = st.session_state.get("token")
        if not token:
            st.error("ログイン情報が見つかりません。再ログインしてください")
            set_page("login")
            safe_rerun()
            st.stop()
        if not prompt or prompt.strip() == "":
            st.error("プロンプトを入力してください")
            st.stop()
        try:
            with st.spinner("生成リクエスト送信中...しばらくお待ちください"):
                content, ct = synth_image(prompt=prompt, token=token, n=n, size=size, prompt_extend=prompt_extend,
                                          watermark=watermark, api_key=api_key_input)
        except Exception as e:
            st.error(f"生成失敗: {e}")
            st.stop()

        if ct and ct.startswith("image/"):
            st.success("生成完了（1枚）")
            st.image(content, use_column_width=True)

            ext = "png"
            if "jpeg" in ct:
                ext = "jpg"
            st.download_button(f"画像をダウンロード（.{ext}）", data=content, file_name=f"synth.{ext}", mime=ct)
        elif ct == "application/zip" or (not ct and content[:2] == b'PK'):
            st.success("生成完了（複数枚 ZIP圧縮）")
            st.download_button("ZIPをダウンロード", data=content, file_name="synth_images.zip", mime="application/zip")
        else:
            try:
                st.image(content, use_column_width=True)
                st.download_button("画像をダウンロード", data=content, file_name="synth.png", mime="image/png")
            except Exception:
                st.write("不明な形式のデータが返されました。ダウンロードして確認してください：")
                st.download_button("データをダウンロード", data=content, file_name="synth.bin",
                                   mime="application/octet-stream")

    if st.button("メインメニューに戻る"):
        set_page("main")
        safe_rerun()


def page_about():
    st.header("このアプリについて")
    st.write("""
    Streamlit フロントエンド + FastAPI バックエンド + EasyOCR + Dashscope + MODNet で構成されたシステムです。  

    **主な機能：** - **登録 / ログイン**: JWTによる認証  
    - **OCR**: 画像からテキストを抽出・表示  
    - **背景除去**: MODNet ONNXモデルを使用した人物等の切り抜き  
    - **切り抜き**: 座標（％）指定による画像のクロップ  
    - **解像度変更**: 倍率またはピクセル指定でのリサイズとフォーマット変換  
    - **画像生成**: Dashscope API（qwen-image）を使用したText-to-Image生成
    """)
    if st.button("メインメニューに戻る"):
        set_page("main")
        safe_rerun()


# ---------- ルーティング処理 ----------
page = st.session_state.get("page", "login")

# 未ログイン時のアクセス制限
if page not in ("about", "login", "register") and not st.session_state.get("token"):
    st.warning("先にログインしてください")
    set_page("login")
    safe_rerun()
    st.stop()

# ページ分岐
if page == "login":
    page_login()
elif page == "register":
    page_register()
elif page == "main":
    page_main()
elif page == "ocr":
    page_ocr()
elif page == "removebg":
    page_removebg()
elif page == "crop":
    page_crop()
elif page == "resize":
    page_resize()
elif page == "synth":
    page_synth()
elif page == "about":
    page_about()
else:
    st.write("不明なページです")



