# uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000

import io
import os
import time
import zipfile
from http import HTTPStatus
from urllib.parse import urlparse, unquote
from pathlib import PurePosixPath
from typing import List, Optional

import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import easyocr

try:
    from dashscope import ImageSynthesis
except Exception:
    ImageSynthesis = None

from sqlalchemy import Column, Integer, String, DateTime, create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from passlib.context import CryptContext
import jwt

try:
    import onnxruntime as ort
except Exception:
    ort = None

# 設定
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./ocr_auth.db")
SECRET_KEY = os.environ.get("SECRET_KEY", "please-change-this-secret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1日

# ONNXモデルのパス設定
DEFAULT_MODNET_ONNX = os.environ.get("MODNET_ONNX_PATH",
                                     "C:/Users/27826/Desktop/project/pretrained/modnet_photographic_portrait_matting.onnx")
# CUDAを使用するかどうか（デフォルトはTrue）
USE_CUDA_FOR_ONNX = os.environ.get("MODNET_USE_CUDA", "1") != "0"

# SQLAlchemyの初期化
# SQLiteの場合はcheck_same_threadをFalseに設定する
engine = create_engine(DATABASE_URL,
                       connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(150), unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=None)


# データベースのテーブル作成
Base.metadata.create_all(bind=engine)

# パスワードハッシュ化コンテキスト
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# FastAPIアプリケーションの定義
app = FastAPI(title="OCR Service with Auth + MODNet + Image Synthesis (Dashscope)")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# EasyOCRの初期化
# GPU=Falseに設定していますが、環境に応じてTrueに変更可能です
try:
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
except Exception as e:
    raise RuntimeError(f"EasyOCRの初期化に失敗しました: {e}")

# セキュリティ依存関係 (Bearer Token)
bearer = HTTPBearer()

# ========== MODNet ONNX 読み込みと推論ツール ==========
ONNX_SESSION = None
ONNX_INPUT_NAME = None
ONNX_OUTPUT_NAME = None


def _get_onnx_providers():
    """
    利用可能なONNX Runtimeプロバイダー（CPU/CUDA）を取得します。
    """
    if ort is None:
        return ["CPUExecutionProvider"]
    try:
        available = ort.get_all_providers()
    except Exception:
        try:
            available = ort.get_available_providers()
        except Exception:
            available = []
    providers = []
    if USE_CUDA_FOR_ONNX and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    if not providers:
        providers = None
    return providers


@app.on_event("startup")
def load_modnet_onnx():
    """
    アプリケーション起動時にMODNetのONNXモデルをロードします。
    """
    global ONNX_SESSION, ONNX_INPUT_NAME, ONNX_OUTPUT_NAME
    if ort is None:
        print(
            "警告: onnxruntimeがインストールされていません。MODNet ONNX推論は利用できません。onnxruntimeまたはonnxruntime-gpuをインストールしてください。")
        return

    model_path = DEFAULT_MODNET_ONNX
    if not os.path.exists(model_path):
        print(f"MODNet ONNXモデルが見つかりません: {model_path}")
        print(
            "事前学習済みの modnet_photographic_portrait_matting.onnx を配置するか、MODNET_ONNX_PATH環境変数を設定してください。")
        ONNX_SESSION = None
        return

    providers = _get_onnx_providers()
    try:
        sess_opts = ort.SessionOptions()
        if providers is None:
            ONNX_SESSION = ort.InferenceSession(model_path, sess_options=sess_opts)
        else:
            ONNX_SESSION = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)
        inputs = ONNX_SESSION.get_inputs()
        outputs = ONNX_SESSION.get_outputs()
        if len(inputs) > 0:
            ONNX_INPUT_NAME = inputs[0].name
        if len(outputs) > 0:
            ONNX_OUTPUT_NAME = outputs[0].name
        print(f"MODNet ONNXモデルをロードしました: {model_path}. Providers: {ONNX_SESSION.get_providers()}")
    except Exception as e:
        print(f"MODNet ONNXモデルのロードに失敗しました: {e}")
        ONNX_SESSION = None


def _pad_to_multiple_of_32(img: Image.Image):
    """
    画像を32の倍数になるようにパディングします（MODNetの入力要件）。
    戻り値: (パディング後の画像, (left, top, pad_w, pad_h))
    """
    w, h = img.size
    pad_w = (32 - (w % 32)) % 32
    pad_h = (32 - (h % 32)) % 32
    if pad_w == 0 and pad_h == 0:
        return img, (0, 0, 0, 0)
    padded = Image.new("RGB", (w + pad_w, h + pad_h), (0, 0, 0))
    padded.paste(img, (0, 0))
    return padded, (0, 0, pad_w, pad_h)


def modnet_remove_background_pil(pil_img: Image.Image, long_side: int = 1024, trimap=None):
    """
    MODNetを使用して画像の背景を除去します。
    1. 前処理（リサイズ、パディング、正規化）
    2. ONNX推論
    3. 後処理（Alphaチャンネルの生成と結合）
    """
    if ONNX_SESSION is None:
        raise RuntimeError(
            "MODNet ONNX sessionがロードされていないため、背景除去を実行できません（MODNET_ONNX_PATHを確認し、onnxruntime(-gpu)をインストールしてください）。")

    # 画像をRGBに変換し、EXIF情報を考慮して回転させる
    img = ImageOps.exif_transpose(pil_img).convert("RGB")
    orig_w, orig_h = img.size

    # 推論速度向上のため、長辺を指定サイズにリサイズ
    scale = 1.0
    if max(orig_w, orig_h) > long_side:
        scale = long_side / float(max(orig_w, orig_h))
    target_w = max(1, int(round(orig_w * scale)))
    target_h = max(1, int(round(orig_h * scale)))
    resized = img.resize((target_w, target_h), Image.LANCZOS)

    # 32の倍数にパディング
    padded, pad = _pad_to_multiple_of_32(resized)
    pw, ph = padded.size

    # 配列化と正規化
    arr = np.array(padded).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # CHW形式へ変換
    arr = np.expand_dims(arr, 0).astype(np.float32)  # バッチ次元を追加

    # 推論実行
    feed = {ONNX_INPUT_NAME: arr}
    try:
        outs = ONNX_SESSION.run(None, feed)
    except Exception as e:
        raise RuntimeError(f"ONNX推論に失敗しました: {e}")

    # Alphaマットの抽出
    alpha = None
    out = outs[0]
    if out.ndim == 4:
        alpha = out[0, 0, :, :]
    elif out.ndim == 3:
        if out.shape[0] == 1:
            alpha = out[0, :, :]
        else:
            alpha = np.squeeze(out)
    elif out.ndim == 2:
        alpha = out
    else:
        alpha = np.squeeze(out)

    # クリップと画像化
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha_img = Image.fromarray((alpha * 255.0).astype(np.uint8))

    # パディング除去と元のサイズへの復元
    if alpha_img.size != (pw, ph):
        alpha_img = alpha_img.resize((pw, ph), Image.LANCZOS)
    if pad != (0, 0, 0, 0):
        right = pw - pad[2]
        bottom = ph - pad[3]
        alpha_img = alpha_img.crop((0, 0, right, bottom))
    if alpha_img.size != (target_w, target_h):
        alpha_img = alpha_img.resize((target_w, target_h), Image.LANCZOS)
    if (target_w, target_h) != (orig_w, orig_h):
        alpha_img = alpha_img.resize((orig_w, orig_h), Image.LANCZOS)

    # 元画像にAlphaチャンネルを追加
    rgba = ImageOps.exif_transpose(pil_img).convert("RGBA")
    alpha_rgba = alpha_img.convert("L")
    rgba.putalpha(alpha_rgba)
    return rgba


# ========== END MODNet 部分 ==========

# Pydanticモデル
class OCRLine(BaseModel):
    text: str
    confidence: float
    bbox: List[List[float]]


class OCRResponse(BaseModel):
    lines: List[OCRLine]
    raw_text: str


class UserCreate(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# 画像生成リクエストモデル
class ImageSynthRequest(BaseModel):
    prompt: str
    n: Optional[int] = 1
    size: Optional[str] = "1328*1328"
    prompt_extend: Optional[bool] = True
    watermark: Optional[bool] = True
    # フロントエンドからのテスト用APIキー（オプション、環境変数を優先）
    api_key: Optional[str] = None


# DBヘルパー関数
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()


def create_user(db: Session, username: str, password: str) -> User:
    hashed = pwd_context.hash(password)
    user = User(username=username, hashed_password=hashed)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[int] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = int(time.time() + expires_delta * 60)
    else:
        expire = None
    # デフォルトの有効期限を使用
    from datetime import datetime, timedelta
    if expire is None:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Tokenの有効期限が切れています")
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="無効なTokenです")


def get_current_username(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    token = credentials.credentials
    payload = decode_access_token(token)
    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=401, detail="Tokenにユーザー情報が含まれていません")
    return username


# 登録とログインのインターフェース
@app.post("/register", response_model=TokenResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    if get_user_by_username(db, user.username):
        raise HTTPException(status_code=400, detail="ユーザー名は既に存在します")
    new_user = create_user(db, user.username, user.password)
    access_token = create_access_token({"sub": new_user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/login", response_model=TokenResponse)
def login(user: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user_by_username(db, user.username)
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="ユーザー名またはパスワードが間違っています")
    access_token = create_access_token({"sub": db_user.username})
    return {"access_token": access_token, "token_type": "bearer"}


# === OCR ロジック ===
def preprocess_pil_image(pil_img: Image.Image, max_size: int = 1600):
    """
    OCR認識率向上のため、画像を適切なサイズにリサイズします。
    """
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    w, h = pil_img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        pil_img = pil_img.resize(new_size, Image.LANCZOS)
    return pil_img


@app.post("/ocr", response_model=OCRResponse)
async def ocr_image(file: UploadFile = File(...), username: str = Depends(get_current_username)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="ファイルがアップロードされていません")
    contents = await file.read()
    try:
        pil = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {e}")

    # 前処理
    proc = preprocess_pil_image(pil, max_size=1600)
    proc_w, proc_h = proc.size
    img_np = np.array(proc)

    try:
        raw_results = reader.readtext(img_np, detail=1, paragraph=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR認識に失敗しました: {e}")

    lines = []
    raw_texts = []
    for bbox, text, conf in raw_results:
        # 座標を正規化（0.0〜1.0）
        norm_bbox = [[float(x) / proc_w, float(y) / proc_h] for (x, y) in bbox]
        lines.append({"text": text, "confidence": float(conf), "bbox": norm_bbox})
        raw_texts.append(text)

    return {"lines": lines, "raw_text": "\n".join(raw_texts)}


# === 背景除去機能 ===
@app.post("/remove_bg")
async def remove_bg_endpoint(file: UploadFile = File(...), username: str = Depends(get_current_username)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="ファイルがアップロードされていません")
    content = await file.read()
    try:
        pil = Image.open(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {e}")

    if ONNX_SESSION is None:
        raise HTTPException(status_code=500,
                            detail=f"MODNetモデルがロードされていません（{DEFAULT_MODNET_ONNX} の存在と onnxruntime(-gpu) のインストールを確認してください）")

    try:
        # 環境変数から長辺のサイズを取得（デフォルト1024）
        long_side = int(os.environ.get("MODNET_LONG_SIDE", "1024"))
        result = modnet_remove_background_pil(pil, long_side=long_side)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"背景除去に失敗しました: {e}")

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


# === 切り抜き（クロップ）機能 ===
@app.post("/crop")
async def crop_image_endpoint(
        file: UploadFile = File(...),
        x1: float = Form(...),
        y1: float = Form(...),
        x2: float = Form(...),
        y2: float = Form(...),
        username: str = Depends(get_current_username),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="ファイルがアップロードされていません")
    content = await file.read()
    try:
        pil = Image.open(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {e}")

    img = ImageOps.exif_transpose(pil).convert("RGBA")
    w, h = img.size

    # 相対座標（0.0-1.0）または絶対座標をピクセル値に変換
    def to_px(val, maxv):
        try:
            v = float(val)
        except Exception:
            raise HTTPException(status_code=400, detail="切り抜き座標は数値である必要があります")
        if 0.0 <= v <= 1.0:
            return int(round(v * maxv))
        return int(round(v))

    left = to_px(x1, w)
    top = to_px(y1, h)
    right = to_px(x2, w)
    bottom = to_px(y2, h)

    # 範囲チェック
    left = max(0, min(left, w - 1))
    right = max(0, min(right, w))
    top = max(0, min(top, h - 1))
    bottom = max(0, min(bottom, h))

    if right <= left or bottom <= top:
        raise HTTPException(status_code=400,
                            detail="無効な切り抜き領域です（x2/x1 または y2/y1 の順序が間違っているか、サイズが0です）")

    try:
        cropped = img.crop((left, top, right, bottom))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"切り抜きに失敗しました: {e}")

    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


# === 解像度変更 / 画質調整 ===
@app.post("/resize")
async def resize_image_endpoint(
        file: UploadFile = File(...),
        mode: str = Form(...),
        scale: Optional[float] = Form(None),
        width: Optional[int] = Form(None),
        height: Optional[int] = Form(None),
        out_format: str = Form("png"),
        quality: Optional[int] = Form(90),
        username: str = Depends(get_current_username),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="ファイルがアップロードされていません")
    content = await file.read()
    try:
        pil = Image.open(io.BytesIO(content)).convert("RGBA")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {e}")

    w, h = pil.size

    if mode not in ("scale", "dims"):
        raise HTTPException(status_code=400, detail="modeは 'scale' または 'dims' である必要があります")

    # 倍率指定モード
    if mode == "scale":
        if scale is None:
            raise HTTPException(status_code=400, detail="mode=scale の場合は scale を指定してください")
        try:
            s = float(scale)
        except Exception:
            raise HTTPException(status_code=400, detail="scale は数値である必要があります")
        if s <= 0 or s > 10:
            raise HTTPException(status_code=400, detail="scale が適切な範囲(0..10)を超えています")
        new_w = max(1, int(round(w * s)))
        new_h = max(1, int(round(h * s)))
    # 寸法指定モード
    else:
        if width is None and height is None:
            raise HTTPException(status_code=400, detail="mode=dims の場合は width または height を指定してください")
        if width is None:
            try:
                h_target = int(height)
            except Exception:
                raise HTTPException(status_code=400, detail="height は整数である必要があります")
            scale_h = h_target / float(h)
            new_h = max(1, h_target)
            new_w = max(1, int(round(w * scale_h)))
        elif height is None:
            try:
                w_target = int(width)
            except Exception:
                raise HTTPException(status_code=400, detail="width は整数である必要があります")
            scale_w = w_target / float(w)
            new_w = max(1, int(w_target))
            new_h = max(1, int(round(h * scale_w)))
        else:
            new_w = max(1, int(width))
            new_h = max(1, int(height))

    try:
        resized = pil.resize((new_w, new_h), Image.LANCZOS)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"サイズ変更に失敗しました: {e}")

    buf = io.BytesIO()
    fmt = out_format.lower()
    if fmt == "jpeg" or fmt == "jpg":
        out_img = resized.convert("RGB")
        q = int(quality) if quality is not None else 90
        q = max(1, min(95, q))
        out_img.save(buf, format="JPEG", quality=q, optimize=True)
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/jpeg")
    else:
        resized.save(buf, format="PNG", optimize=True)
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")


# === 画像生成 (Image Synthesis) ===
@app.post("/synth_image")
async def synth_image(req: ImageSynthRequest, username: str = Depends(get_current_username)):
    """
    テキストから画像を生成するインターフェース（JWT保護付き）
    リクエスト例(JSON):
    {
        "prompt": "...",
        "n": 1,
        "size": "1328*1328",
        "prompt_extend": true,
        "watermark": true,
        "api_key": null   # オプション: テスト用、通常は環境変数 DASHSCOPE_API_KEY を使用
    }
    レスポンス:
      - n == 1: 画像のバイト列を直接返す (image/png または image/jpeg)
      - n > 1: 複数の画像を含む application/zip を返す
    """
    if ImageSynthesis is None:
        raise HTTPException(status_code=500,
                            detail="dashscope SDKがインストールされていないか利用できません（pip install dashscopeを実行し、利用可能か確認してください）")

    prompt = req.prompt
    n = max(1, int(req.n or 1))
    size = req.size or "1328*1328"
    prompt_extend = bool(req.prompt_extend)
    watermark = bool(req.watermark)

    api_key = os.getenv("DASHSCOPE_API_KEY") or req.api_key
    if not api_key:
        raise HTTPException(status_code=400,
                            detail="サーバー側で DASHSCOPE_API_KEY が設定されておらず、リクエストにも api_key が含まれていません")

    # Dashscope ImageSynthesis の呼び出し
    try:
        rsp = ImageSynthesis.call(
            api_key=api_key,
            model="qwen-image",
            prompt=prompt,
            n=n,
            size=size,
            prompt_extend=prompt_extend,
            watermark=watermark
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ImageSynthesisの呼び出しに失敗しました: {e}")

    if rsp.status_code != HTTPStatus.OK:
        code = getattr(rsp, "code", None)
        message = getattr(rsp, "message", None)
        raise HTTPException(status_code=500,
                            detail=f"画像サービスがエラーを返しました: status={rsp.status_code}, code={code}, message={message}")

    # 結果の解析と画像のダウンロード
    results = []
    try:
        # 複数の結果をサポート
        output = getattr(rsp, "output", None)

        results = getattr(output, "results", []) if output is not None else []
    except Exception:
        results = []

    if not results:
        raise HTTPException(status_code=500, detail="画像サービスは応答しましたが、結果が含まれていません")

    # 各結果の画像をダウンロード
    images_bytes = []
    filenames = []
    for i, result in enumerate(results):
        # result.url から取得
        url = getattr(result, "url", None)
        if not url:
            continue
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            content = r.content
            file_name = PurePosixPath(unquote(urlparse(url).path)).parts[-1]
            if not file_name:
                file_name = f"image_{i + 1}.png"
            images_bytes.append(content)
            filenames.append(file_name)
        except Exception as e:
            print(f"生成画像のダウンロードに失敗しました（{url}）: {e}")

    if not images_bytes:
        raise HTTPException(status_code=500, detail="生成された画像をダウンロードできませんでした")

    if len(images_bytes) == 1:
        img_bytes = images_bytes[0]
        ext = os.path.splitext(filenames[0])[1].lower()
        mime = "image/png"
        if ext in (".jpg", ".jpeg"):
            mime = "image/jpeg"
        return StreamingResponse(io.BytesIO(img_bytes), media_type=mime)

    # 複数画像の場合はZIPに圧縮
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, b in zip(filenames, images_bytes):
            zname = name
            if zname in zf.namelist():
                base, ext = os.path.splitext(zname)
                zname = f"{base}_{len(zf.namelist())}{ext}"
            zf.writestr(zname, b)
    zip_buf.seek(0)
    return StreamingResponse(zip_buf, media_type="application/zip")


from datetime import datetime


@app.get("/health")
def health():
    providers = None
    try:
        if ONNX_SESSION is not None:
            providers = ONNX_SESSION.get_providers()
    except Exception:
        providers = None
    return {"status": "ok", "time": datetime.utcnow().isoformat(), "modnet_providers": providers}