"""
streamlit_app.py
Adapted Streamlit deployment for GitHub with file structure:
 - streamlit_app.py in root
 - model files in _MODELLING/ directory
 - looks for combined checkpoint 'yolov12_dqn_classifier.pth'
 - reads metadata 'yolov12_dqn_metadata.json' if present
 - supports YOLO detection, DQN refinement, embedder, and TF classifier

Run with: streamlit run streamlit_app.py
"""
import random
import os
import json
import time
from pathlib import Path
from typing import Optional, Tuple
import streamlit as st
import numpy as np
from PIL import Image

# try imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None
    nn = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import tensorflow as tf
except Exception:
    tf = None

# ----------------- Config -----------------
# Set MODELLING_ROOT relative to script location
SCRIPT_DIR = Path(__file__).parent.absolute()
MODELLING_ROOT = os.environ.get("MODELLING_ROOT", str(SCRIPT_DIR / "_MODELLING"))
os.makedirs(MODELLING_ROOT, exist_ok=True)

st.set_page_config(page_title="YOLOv12 + DQN Refine Demo", layout="wide")
st.title("YOLOv12 + DQN Refinement + TF Classifier")

# ----------------- Small utilities -----------------
def to_contig_np(x):
    x = np.asarray(x)
    if not x.flags['C_CONTIGUOUS']:
        x = np.ascontiguousarray(x)
    return x

def pil_to_rgb(img):
    if isinstance(img, str):
        img = Image.open(img)
    return img.convert('RGB')

def preprocess_image_to_tensor(img_or_path, img_size=(128,128), device=None, normalize=True):
    img = img_or_path
    if isinstance(img_or_path, str):
        img = Image.open(img_or_path).convert('RGB')
    elif isinstance(img_or_path, Image.Image):
        img = img_or_path.convert('RGB')
    else:
        img = Image.fromarray(to_contig_np(img_or_path).astype('uint8')).convert('RGB')
    
    img = img.resize(img_size, resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = to_contig_np(arr)
    
    if torch is None:
        return arr.transpose(2,0,1)[None]
    
    t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float()
    if device is not None:
        t = t.to(device)
    if normalize and torch is not None:
        mean = torch.tensor([0.485,0.456,0.406], device=t.device).view(1,3,1,1)
        std = torch.tensor([0.229,0.224,0.225], device=t.device).view(1,3,1,1)
        t = (t - mean) / std
    return t

# ----------------- Inference helpers -----------------
def infer_emb_dim(emb_state):
    if emb_state is None:
        return 256
    actual = emb_state
    if isinstance(emb_state, dict):
        for key in ['model', 'embedder_state_dict', 'state_dict', 'embedder']:
            if key in emb_state and isinstance(emb_state[key], dict):
                actual = emb_state[key]
                break
    if isinstance(actual, dict) and 'proj.3.weight' in actual:
        try:
            return actual['proj.3.weight'].shape[0]
        except Exception:
            pass
    return 256

def infer_dqn_params(dqn_state):
    if dqn_state is None:
        return None, None, None
    actual = dqn_state
    if isinstance(dqn_state, dict):
        for key in ['net_state_dict', 'dqn_state_dict', 'agent_state_dict', 'q_state_dict', 'state_dict', 'model_state_dict', 'net']:
            if key in dqn_state and isinstance(dqn_state[key], dict):
                actual = dqn_state[key]
                break
    if not isinstance(actual, dict):
        return None, None, None
    
    obs_dim = None
    hidden = None
    n_actions = None
    
    if '0.weight' in actual:
        try:
            shape = actual['0.weight'].shape
            hidden = shape[0]
            obs_dim = shape[1]
        except Exception:
            pass
    
    if '8.weight' in actual:
        try:
            n_actions = actual['8.weight'].shape[0]
        except Exception:
            pass
    
    return obs_dim, hidden, n_actions

# ----------------- Model building blocks -----------------
if torch is not None:
    class SmallConvEmbedder(nn.Module):
        def __init__(self, emb_dim=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
            )
            self.proj = nn.Sequential(
                nn.Linear(128, max(emb_dim,64)),
                nn.ReLU(),
                nn.LayerNorm(max(emb_dim,64)),
                nn.Linear(max(emb_dim,64), emb_dim),
                nn.ReLU()
            )
        
        def forward(self, x):
            z = self.net(x)
            z = z.view(z.size(0), -1)
            z = self.proj(z)
            return z
else:
    SmallConvEmbedder = None

class YoloFeatureExtractor:
    def __init__(self, yolo_obj=None, emb_dim=256, crop_size=(128,128), device=None):
        self.yolo = yolo_obj
        self.device = device if device is not None else (torch.device('cuda') if torch is not None and torch.cuda.is_available() else (torch.device('cpu') if torch is not None else None))
        self.crop_size = tuple(crop_size)
        
        if torch is not None and SmallConvEmbedder is not None:
            self.embed_net = SmallConvEmbedder(emb_dim=emb_dim).to(self.device)
        else:
            self.embed_net = None
        self.emb_dim = emb_dim
    
    def detect_top1(self, img):
        """Return bbox [x1,y1,x2,y2], conf, cls or (None,0,None)."""
        if self.yolo is None or YOLO is None:
            return None, 0.0, None
        try:
            preds = self.yolo.predict(source=img, imgsz=max(self.crop_size), verbose=False)
        except Exception:
            preds = self.yolo(img)
        
        if len(preds) == 0:
            return None, 0.0, None
        
        r = preds[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None, 0.0, None
        
        confs = [float(b.conf) for b in boxes]
        idx = int(np.argmax(confs))
        b = boxes[idx]
        xyxy = b.xyxy.cpu().numpy().reshape(-1).tolist()
        conf = float(b.conf)
        cls = int(b.cls) if hasattr(b, "cls") else None
        return [int(v) for v in xyxy], conf, cls
    
    def embed_crop(self, pil_crop):
        if self.embed_net is None or torch is None:
            return None
        t = preprocess_image_to_tensor(pil_crop, img_size=self.crop_size, device=self.device)
        with torch.no_grad():
            emb = self.embed_net(t).squeeze(0)
        return emb.cpu().numpy().astype('float32')
    
    def get_obs_for_bbox(self, pil_img, bbox):
        x1,y1,x2,y2 = [int(v) for v in bbox]
        crop = pil_img.crop((x1,y1,x2,y2)).resize(self.crop_size)
        emb = self.embed_crop(crop)
        W,H = pil_img.size
        norm = np.array([x1/float(W), y1/float(H), x2/float(W), y2/float(H)], dtype='float32')
        conf = np.array([0.0], dtype='float32')
        if emb is None:
            emb = np.zeros(self.emb_dim, dtype='float32')
        return np.concatenate([emb, norm, conf], axis=0)

GAMMA = 0.95

# DQNAgent
if torch is not None:
    class DQNAgent:
        def __init__(self, obs_dim, n_actions, hidden=192, lr=5e-5, device=None, weight_decay=5e-4, dropout=0.3, seed=42):
            self.device = device
            self.obs_dim = obs_dim
            self.n_actions = n_actions
            self.hidden = hidden
            torch.manual_seed(seed)
            
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.ReLU(),
                nn.LayerNorm(hidden),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.LayerNorm(hidden // 2),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, n_actions)
            ).to(self.device)
            
            import copy
            self.target = copy.deepcopy(self.net)
            self.opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
            self.gamma = GAMMA
        
        def act(self, obs, eps=0.0):
            if random.random() < eps:
                return random.randrange(self.n_actions)
            
            obs_np = np.asarray(obs, dtype=np.float32)
            if obs_np.ndim == 1:
                obs_np = obs_np
            elif obs_np.ndim == 2 and obs_np.shape[0] == 1:
                obs_np = obs_np[0]
            else:
                obs_np = obs_np.ravel()
            
            if torch is None:
                return int(np.argmax(np.zeros(self.n_actions)))
            
            t = torch.from_numpy(obs_np.astype('float32')).unsqueeze(0).to(self.device)
            
            expected_in = None
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    expected_in = m.in_features
                    break
            if expected_in is None:
                expected_in = t.shape[1]
            
            if t.shape[1] < expected_in:
                pad = torch.zeros((t.shape[0], expected_in - t.shape[1]), device=t.device, dtype=t.dtype)
                t = torch.cat([t, pad], dim=1)
            elif t.shape[1] > expected_in:
                t = t[:, :expected_in]
            
            with torch.no_grad():
                q = self.net(t)[0].cpu().numpy()
            return int(np.argmax(q))
else:
    DQNAgent = None

# ----------------- Loading helpers -----------------
@st.cache_resource
def load_deployment_meta(mod_root: str):
    preferred = Path(mod_root) / 'yolov12_dqn_metadata.json'
    if preferred.exists():
        try:
            return json.load(open(preferred, 'r'))
        except Exception:
            pass
    
    meta_files = sorted([p for p in Path(mod_root).glob('deployment_meta*.json')])
    if meta_files:
        try:
            return json.load(open(meta_files[-1], 'r'))
        except Exception:
            return {}
    return {}

def _try_load_torch(path):
    if torch is None:
        return None
    try:
        return torch.load(path, map_location='cpu')
    except Exception:
        return None

@st.cache_resource
def load_models(mod_root: str):
    """
    Return dict with keys: yolo, fe, dqn, tf_clf, meta, class_map, inv_class_map
    """
    meta = load_deployment_meta(mod_root) or {}
    models = {'meta': meta, 'yolo': None, 'fe': None, 'dqn': None, 'tf_clf': None, 'class_map': None, 'inv_class_map': None}
    
    # TF classifier
    tf_path = os.path.join(mod_root, meta.get('saved_files', {}).get('tf_clf', 'clf_best.h5'))
    if tf is not None and os.path.exists(tf_path):
        try:
            models['tf_clf'] = tf.keras.models.load_model(tf_path, compile=False)
        except Exception as e:
            st.warning(f"Failed loading TF classifier: {e}")
    
    # YOLO - look for yolov12x-cls.pt in _MODELLING
    yolo_path = os.path.join(mod_root, 'yolov12x-cls.pt')
    if YOLO is not None and os.path.exists(yolo_path):
        try:
            models['yolo'] = YOLO(yolo_path)
        except Exception as e:
            st.warning(f"Failed to load YOLO from {yolo_path}: {e}")
    else:
        # try any .pt file
        if YOLO is not None:
            candidates = list(Path(mod_root).glob('*.pt'))
            if candidates:
                try:
                    models['yolo'] = YOLO(str(candidates[-1]))
                except Exception as e:
                    st.warning(f"Candidate YOLO load failed: {e}")
    
    # Combined checkpoint
    combined_path = os.path.join(mod_root, 'yolov12_dqn_classifier.pth')
    ck = None
    if os.path.exists(combined_path) and torch is not None:
        ck = _try_load_torch(combined_path)
        if ck is not None:
            # class_map
            if 'class_to_idx' in ck and isinstance(ck['class_to_idx'], dict):
                try:
                    class_map = {str(k): int(v) for k,v in ck['class_to_idx'].items()}
                    inv_map = {int(v): str(k) for k,v in class_map.items()}
                    models['class_map'] = class_map
                    models['inv_class_map'] = inv_map
                    models['meta'].setdefault('class_map', class_map)
                except Exception:
                    pass
            
            # embedder state
            emb_state = None
            for k in ('embedder_state_dict', 'embedder', 'fe_state_dict', 'fe_state'):
                if k in ck:
                    emb_state = ck[k]
                    break
            
            # dqn state
            dqn_state = None
            for k in ('net_state_dict', 'dqn_state_dict', 'agent_state_dict', 'q_state_dict', 'state_dict'):
                if k in ck:
                    dqn_state = ck[k]
                    break
            
            # infer dimensions
            obs_dim = None
            n_actions = 9
            hidden = 256
            
            if 'obs_dim' in ck:
                try:
                    obs_dim = int(ck['obs_dim'])
                except Exception:
                    pass
            if 'n_actions' in ck:
                try:
                    n_actions = int(ck['n_actions'])
                except Exception:
                    pass
            
            if isinstance(models['meta'], dict):
                if 'obs_dim' in models['meta'] and obs_dim is None:
                    try:
                        obs_dim = int(models['meta']['obs_dim'])
                    except Exception:
                        pass
            
            # Create feature extractor
            try:
                device = torch.device('cpu')
                inf_emb_dim = infer_emb_dim(emb_state)
                emb_dim = inf_emb_dim if inf_emb_dim is not None else 256
                
                fe = YoloFeatureExtractor(yolo_obj=models['yolo'], emb_dim=emb_dim, crop_size=(128,128), device=device)
                if emb_state is not None and fe.embed_net is not None:
                    try:
                        fe.embed_net.load_state_dict(emb_state)
                        fe.embed_net.eval()
                        models['fe'] = fe
                    except Exception:
                        if isinstance(emb_state, dict) and 'model' in emb_state:
                            try:
                                fe.embed_net.load_state_dict(emb_state['model'])
                                fe.embed_net.eval()
                                models['fe'] = fe
                            except Exception:
                                models['fe'] = fe
                        else:
                            models['fe'] = fe
                else:
                    models['fe'] = fe
            except Exception as e:
                st.warning(f"Failed to setup embedder: {e}")
            
            # Create DQN
            if dqn_state is not None:
                inf_obs_dim, inf_hidden, inf_n_actions = infer_dqn_params(dqn_state)
                if inf_obs_dim is not None:
                    obs_dim = inf_obs_dim
                if inf_hidden is not None:
                    hidden = inf_hidden
                if inf_n_actions is not None:
                    n_actions = inf_n_actions
                
                if obs_dim is None and models.get('fe') is not None:
                    obs_dim = models['fe'].emb_dim + 4 + 1
                
                try:
                    if DQNAgent is not None and obs_dim is not None:
                        dqn = DQNAgent(obs_dim, n_actions, hidden=hidden, device=torch.device('cpu'))
                        try:
                            dqn.net.load_state_dict(dqn_state)
                        except Exception:
                            if isinstance(dqn_state, dict) and 'net_state_dict' in dqn_state:
                                dqn.net.load_state_dict(dqn_state['net_state_dict'])
                            elif isinstance(dqn_state, dict) and 'model_state_dict' in dqn_state:
                                dqn.net.load_state_dict(dqn_state['model_state_dict'])
                            else:
                                try:
                                    model_state = {k:v for k,v in dqn_state.items() if k in dqn.net.state_dict()}
                                    dqn.net.load_state_dict(model_state, strict=False)
                                except Exception:
                                    pass
                        dqn.net.eval()
                        models['dqn'] = dqn
                except Exception as e:
                    st.warning(f"Failed to restore DQN: {e}")
    
    # Fallback to separate files if needed
    if models['fe'] is None:
        emb_path = os.path.join(mod_root, 'embedder_state.pth')
        if os.path.exists(emb_path) and torch is not None:
            emb_ck = _try_load_torch(emb_path)
            if emb_ck is not None:
                inf_emb_dim = infer_emb_dim(emb_ck)
                emb_dim = inf_emb_dim if inf_emb_dim is not None else 256
                try:
                    device = torch.device('cpu')
                    fe = YoloFeatureExtractor(yolo_obj=models['yolo'], emb_dim=emb_dim, crop_size=(128,128), device=device)
                    if isinstance(emb_ck, dict):
                        fe.embed_net.load_state_dict(emb_ck)
                    fe.embed_net.eval()
                    models['fe'] = fe
                except Exception as e:
                    st.warning(f"Failed loading embedder_state.pth: {e}")
    
    if models['dqn'] is None:
        dqn_path = os.path.join(mod_root, 'dqn_checkpoint.pth')
        if os.path.exists(dqn_path) and torch is not None:
            ck2 = _try_load_torch(dqn_path)
            if ck2 is not None:
                obs_dim = None
                n_actions = 9
                hidden = 256
                
                if isinstance(ck2, dict):
                    if 'obs_dim' in ck2:
                        try:
                            obs_dim = int(ck2.get('obs_dim'))
                        except Exception:
                            pass
                    if 'n_actions' in ck2:
                        try:
                            n_actions = int(ck2.get('n_actions'))
                        except Exception:
                            pass
                
                inf_obs_dim, inf_hidden, inf_n_actions = infer_dqn_params(ck2)
                if inf_obs_dim is not None:
                    obs_dim = inf_obs_dim
                if inf_hidden is not None:
                    hidden = inf_hidden
                if inf_n_actions is not None:
                    n_actions = inf_n_actions
                
                if obs_dim is None and models.get('fe') is not None:
                    obs_dim = models['fe'].emb_dim + 4 + 1
                
                try:
                    if obs_dim is not None and DQNAgent is not None:
                        dqn = DQNAgent(obs_dim, n_actions, hidden=hidden, device=torch.device('cpu'))
                        try:
                            dqn.net.load_state_dict(ck2)
                        except Exception:
                            for key in ('net_state_dict','state_dict','model_state_dict'):
                                if key in ck2:
                                    try:
                                        dqn.net.load_state_dict(ck2[key])
                                        break
                                    except Exception:
                                        pass
                            else:
                                partial = {k:v for k,v in ck2.items() if k in dqn.net.state_dict()}
                                if partial:
                                    dqn.net.load_state_dict(partial, strict=False)
                        dqn.net.eval()
                        models['dqn'] = dqn
                except Exception as e:
                    st.warning(f"Failed to restore DQN: {e}")
    
    # Ensure class map
    if models['class_map'] is None:
        if isinstance(models['meta'], dict) and 'class_map' in models['meta']:
            try:
                cm = {str(k): int(v) for k, v in models['meta']['class_map'].items()}
                models['class_map'] = cm
                models['inv_class_map'] = {int(v): str(k) for k, v in cm.items()}
            except Exception:
                pass
    
    return models

models = load_models(MODELLING_ROOT)

# ----------------- Sidebar: status -----------------
st.sidebar.header("Model Status")
st.sidebar.write("MODELLING_ROOT:", MODELLING_ROOT)

if models.get('yolo') is not None:
    st.sidebar.success("✅ YOLO: available")
else:
    st.sidebar.info("❌ YOLO: NOT available")

if models.get('dqn') is not None:
    st.sidebar.success("✅ DQN: available")
else:
    st.sidebar.info("❌ DQN: NOT available")

if models.get('fe') is not None:
    st.sidebar.success("✅ Embedder: available")
else:
    st.sidebar.info("❌ Embedder: NOT available")

if models.get('tf_clf') is not None:
    st.sidebar.success("✅ TF classifier: available")
else:
    st.sidebar.info("❌ TF classifier: NOT available")

pseudo = {}
pseudo_path = os.path.join(MODELLING_ROOT, 'pseudo_bboxes.json')
if os.path.exists(pseudo_path):
    try:
        pseudo = json.load(open(pseudo_path, 'r'))
    except Exception:
        pseudo = {}

class_map = models.get('class_map')
inv_class_map = models.get('inv_class_map') or ({int(k):str(v) for k,v in (class_map or {}).items()} if class_map else None)

# ----------------- UI -----------------
col1, col2 = st.columns([1,1])

with col1:
    uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
    example_btn = st.button("Use first pseudo image (if available)")

with col2:
    st.markdown("**Actions**")
    refine_steps = st.slider("Max refine steps (DQN)", min_value=1, max_value=12, value=3)
    run_refine = st.button("Refine & Classify")

image_path = None
img_obj = None

if uploaded is not None:
    tmp_path = os.path.join(MODELLING_ROOT, f"uploaded_{int(time.time())}_{uploaded.name}")
    with open(tmp_path, 'wb') as f:
        f.write(uploaded.getbuffer())
    image_path = tmp_path
    img_obj = Image.open(image_path).convert('RGB')
elif example_btn:
    if len(pseudo) > 0:
        key = list(pseudo.keys())[0]
        if os.path.exists(key):
            image_path = key
        else:
            p = os.path.join(MODELLING_ROOT, key)
            if os.path.exists(p):
                image_path = p
        if image_path:
            img_obj = Image.open(image_path).convert('RGB')
    else:
        st.warning("No pseudo entries available.")

if img_obj is not None:
    st.image(img_obj, caption='Input image', use_container_width=True)

# ----------------- Inference helpers -----------------
def get_initial_bbox_by_yolo(img_path_or_obj, models):
    if models.get('yolo') is None:
        return None
    try:
        if isinstance(img_path_or_obj, str):
            preds = models['yolo'].predict(source=img_path_or_obj, imgsz=640, verbose=False)
        else:
            preds = models['yolo'].predict(source=img_path_or_obj, imgsz=640, verbose=False)
        
        if len(preds) == 0:
            return None
        boxes = getattr(preds[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None
        b = boxes[0]
        xyxy = b.xyxy.cpu().numpy().reshape(-1).tolist()
        return [int(v) for v in xyxy]
    except Exception:
        if models.get('fe') is not None:
            try:
                return models['fe'].detect_top1(img_path_or_obj)[0]
            except Exception:
                return None
        return None

def refine_with_dqn(image_path: str, init_bbox: Tuple[int,int,int,int], models, max_steps=12):
    if models.get('dqn') is None or models.get('fe') is None:
        return init_bbox, None
    env = ActiveCropEnvTorch(image_path, init_bbox, models['fe'], img_size=(128,128), max_steps=max_steps)
    s = env.reset()
    for _ in range(max_steps):
        a = models['dqn'].act(s, eps=0.0)
        s, r, done, info = env.step(a)
        if done:
            break
    return env.bbox, info.get('iou', None)

def classify_crop_with_tf(crop_pil, tf_clf):
    if tf_clf is None:
        return None
    arr = (np.asarray(crop_pil).astype('float32')/255.0)[None]
    try:
        probs = tf_clf.predict(arr, verbose=0)[0]
        return probs
    except Exception as e:
        st.warning(f"TF classifier predict failed: {e}")
        return None

# ActiveCropEnvTorch
class ActiveCropEnvTorch:
    def __init__(self, image_path, target_bbox, fe: YoloFeatureExtractor, img_size=(128,128), max_steps=12):
        self.image_path = image_path
        self.target_bbox = [int(x) for x in target_bbox]
        self.img = pil_to_rgb(image_path)
        self.W, self.H = self.img.size
        self.max_steps = max_steps
        self.fe = fe
        self.img_size = img_size
        self.reset()
    
    def reset(self):
        w = int(self.W * 0.6)
        h = int(self.H * 0.6)
        cx, cy = self.W//2, self.H//2
        self.bbox = [max(0, cx-w//2), max(0, cy-h//2), min(self.W-1, cx+w//2), min(self.H-1, cy+h//2)]
        self.steps = 0
        self.prev_iou = self._iou(self.bbox, self.target_bbox)
        return self._get_obs()
    
    def step(self, action):
        dx = int(self.W*0.05)
        dy = int(self.H*0.05)
        x1,y1,x2,y2 = self.bbox
        
        if action == 0: x1=max(0,x1-dx); x2=max(x1+1,x2-dx)
        elif action == 1: x1=min(self.W-1,x1+dx); x2=min(self.W-1,x2+dx)
        elif action == 2: y1=max(0,y1-dy); y2=max(y1+1,y2-dy)
        elif action == 3: y1=min(self.H-1,y1+dy); y2=min(self.H-1,y2+dy)
        elif action == 4: x1=max(0,x1-dx); x2=min(self.W-1,x2+dx)
        elif action == 5: x1=min(x2-1,x1+dx); x2=max(x1+1,x2-dx)
        elif action == 6: y1=max(0,y1-dy); y2=min(self.H-1,y2+dy)
        elif action == 7: y1=min(y2-1,y1+dy); y2=max(y1+1,y2-dy)
        elif action == 8: pass
        
        self.bbox = [int(x1),int(y1),int(x2),int(y2)]
        self.steps += 1
        iou = self._iou(self.bbox, self.target_bbox)
        reward = (iou - self.prev_iou) * 12.0 - 0.01 + 3.0 * (iou if iou > 0.5 else 0.0)
        
        if (x2-x1) < self.W*0.08 or (y2-y1) < self.H*0.08:
            reward -= 2.0
        
        done = False
        if action == 8:
            done = True
            reward += 12.0 if iou >= 0.5 else -6.0
        if self.steps >= self.max_steps:
            done = True
        
        self.prev_iou = iou
        return self._get_obs(), reward, done, {'iou': iou, 'bbox': self.bbox}
    
    def _get_obs(self):
        return self.fe.get_obs_for_bbox(self.img, self.bbox)
    
    @staticmethod
    def _iou(b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter_w = max(0, x2-x1+1)
        inter_h = max(0, y2-y1+1)
        inter = inter_w * inter_h
        area1 = (b1[2]-b1[0]+1)*(b1[3]-b1[1]+1)
        area2 = (b2[2]-b2[0]+1)*(b2[3]-b2[1]+1)
        union = area1 + area2 - inter
        if union <= 0:
            return 0.0
        return inter/union

# ----------------- Run pipeline -----------------
if run_refine and img_obj is not None:
    st.markdown("### Running detection → refine → classify")
    
    # 1) Initial bbox
    init_bbox = None
    try:
        if uploaded is not None:
            init_bbox = get_initial_bbox_by_yolo(image_path, models)
        else:
            init_bbox = get_initial_bbox_by_yolo(img_obj, models)
    except Exception:
        init_bbox = None
    
    # 2) Fallback to pseudo
    if init_bbox is None and uploaded is not None:
        fname = os.path.basename(image_path)
        if fname in pseudo:
            init_bbox = pseudo[fname].get('bbox')
    
    if init_bbox is None and len(pseudo) > 0:
        try:
            init_bbox = list(pseudo.values())[0].get('bbox')
        except Exception:
            init_bbox = None
    
    if init_bbox is None:
        st.error("No initial bounding box found. Please provide YOLO weights or pseudo_bboxes.json.")
    else:
        st.write("**Initial bbox:**", init_bbox)
        x1,y1,x2,y2 = [int(v) for v in init_bbox]
        crop_init = img_obj.crop((x1,y1,x2,y2)).resize((160,160))
        st.image(crop_init, caption='Initial crop (YOLO/pseudo)', width=220)
        
        # 3) Refine with DQN
        final_bbox, final_iou = refine_with_dqn(image_path if uploaded is not None else image_path or None, init_bbox, models, max_steps=refine_steps)
        st.write("**Refined bbox:**", final_bbox, " | **IoU:**", final_iou)
        x1,y1,x2,y2 = [int(v) for v in final_bbox]
        crop_final = img_obj.crop((x1,y1,x2,y2)).resize((160,160))
        st.image(crop_final, caption='Final crop (after DQN)', width=220)
        
        # 4) Classify with TF
        probs = classify_crop_with_tf(crop_final, models.get('tf_clf'))
        if probs is not None:
            top_idx = int(np.argmax(probs))
            inv = inv_class_map
            label = inv.get(top_idx, str(top_idx)) if inv else str(top_idx)
            st.success(f"**Predicted class:** {label} (idx {top_idx})")
            st.write("**Top probabilities:**")
            topk = np.argsort(probs)[::-1][:10]
            for idx in topk:
                name = inv.get(int(idx), str(idx)) if inv else str(idx)
                st.write(f"  • {name}: {probs[int(idx)]:.4f}")
        else:
            st.info("TF classifier not available — skipping classification.")
        
        # 5) Overlay visualization
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(img_obj)
            ax.axis('off')
            
            ix1,iy1,ix2,iy2 = [int(v) for v in init_bbox]
            fx1,fy1,fx2,fy2 = [int(v) for v in final_bbox]
            
            ax.add_patch(mpatches.Rectangle((ix1,iy1), ix2-ix1, iy2-iy1, 
                                           edgecolor='lime', facecolor='none', 
                                           linewidth=3, label='Initial'))
            ax.add_patch(mpatches.Rectangle((fx1,fy1), fx2-fx1, fy2-fy1, 
                                           edgecolor='red', facecolor='none', 
                                           linewidth=3, label='Refined'))
            ax.legend(loc='upper right', fontsize=12)
            
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not create overlay visualization: {e}")

# ----------------- Footer -----------------
st.sidebar.markdown("---")
st.sidebar.header("Deployment Info")
st.sidebar.write(
    "This app looks for model files in the `_MODELLING/` directory:\n"
    "- `yolov12_dqn_classifier.pth` (combined checkpoint)\n"
    "- `yolov12_dqn_metadata.json` (metadata)\n"
    "- `yolov12x-cls.pt` (YOLO weights)\n"
    "- `clf_best.h5` (TF classifier)\n"
    "- `pseudo_bboxes.json` (fallback bboxes)"
)

st.sidebar.markdown("**Files in MODELLING_ROOT:**")
for f in ['yolov12_dqn_classifier.pth', 'yolov12_dqn_metadata.json', 
          'yolov12x-cls.pt', 'embedder_state.pth', 'dqn_checkpoint.pth', 
          'clf_best.h5', 'pseudo_bboxes.json']:
    exists = os.path.exists(os.path.join(MODELLING_ROOT, f))
    st.sidebar.write(f"{'✅' if exists else '❌'} {f}")

st.markdown("\n---\n*YOLOv12 + DQN Refinement Demo - GitHub Deployment*")