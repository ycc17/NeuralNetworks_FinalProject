# -*- coding: utf-8 -*-
"""
跨資料集評估 — Antwerp 公開資料集【類別加權版】
================================================
資料: antwerp_dataset.csv (比利時安特衛普市公開 benchmark, 13萬筆)
目標: 將 AI-ERA NS-3 模擬訓練好的 CNN+LSTM 遷移到歐洲都市真實量測資料

【本版與基本版的差異】
  Antwerp 資料 SF7 佔 74%, 極度不平衡。基本版微調後模型會掉進
  "多數類別崩塌" (全猜 SF7) -> Acc 看似 73.89% 但 F1 僅 0.628。
  本版在 CrossEntropy 加入【類別權重】(inverse frequency),
  放大少數類 (SF8-SF12) 的錯誤懲罰, 強迫模型學習區辨。
  預期: Acc 可能下降, 但 F1 上升、混淆矩陣對角線浮現 = 真正學會分類。

  報告主指標一律以 F1 為準 (Accuracy 因不平衡會嚴重誤導)。

特徵對齊:
  X-pos<-Longitude  Y-pos<-Latitude
  第3維(RxPw)<-72台最強RSSI   第4維(原SNR)<-收到的基地台數量
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from model_cnnlstm import CNNLSTM

# ============ 設定 ============
CSV_PATH = "./antwerp_dataset.csv"
CKPT_PATH = "./model/LSTM/[CNNLSTM](MLPepoch-10000)-(init_lr-0.0001)-(batch-512)-(layer-1)-(acc-80).pt"
SEQ_LEN = 6
SF_OFFSET = 7
SEED = 42
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_clf_eval(y_true, y_pred, average="weighted"):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, warn_for=tuple())
    return acc, p, r, f1


def per_class_recall(cm):
    """從 normalize='true' 的混淆矩陣取對角線 = 每類 recall。"""
    diag = np.diag(cm)
    return {f"SF{i+7}": round(float(diag[i]), 3) for i in range(len(diag))}


def load_antwerp_dataset():
    """讀取 Antwerp 資料, 做特徵工程, 回傳 (N,6,4) + label(0..5)。"""
    df = pd.read_csv(CSV_PATH)
    bs_cols = [c for c in df.columns if c.startswith("BS")]
    bs = df[bs_cols].values.astype(np.float32)

    mask = bs > -200                                   # -200 = 該基地台沒收到
    best_rssi = np.where(mask.any(axis=1),
                         np.max(np.where(mask, bs, -999), axis=1), -200.0)
    n_gw = mask.sum(axis=1).astype(np.float32)         # 收到的基地台數量

    lat = df["Latitude"].values.astype(np.float32)
    lon = df["Longitude"].values.astype(np.float32)
    lat0, lon0 = lat.min(), lon.min()
    x_pos = (lon - lon0) * 111000.0 * np.cos(np.radians(51.2))
    y_pos = (lat - lat0) * 111000.0

    # 4 特徵順序需與訓練時一致: X-pos, Y-pos, RxPw(best RSSI), SNR(用n_gw替代)
    feats = np.stack([x_pos, y_pos, best_rssi, n_gw], axis=1).astype(np.float32)

    # 特徵標準化 (z-score): 三個特徵尺度差數量級, 不標準化會讓大數值主導梯度
    mu, sigma = feats.mean(axis=0), feats.std(axis=0) + 1e-6
    feats = (feats - mu) / sigma

    sf = df["SF"].values.astype(np.int64)

    # 依時間排序後滑動視窗切序列 (此資料無節點ID, 為簡化視為時序流, 報告需說明)
    order = np.argsort(df["RX Time"].values)
    feats, sf = feats[order], sf[order]

    X, Y = [], []
    for i in range(len(feats) - SEQ_LEN):
        X.append(feats[i:i + SEQ_LEN])
        Y.append(sf[i + SEQ_LEN])
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int64) - SF_OFFSET
    return X, Y


def make_loader(x, y, batch_size, shuffle):
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long).view(-1)
    return DataLoader(TensorDataset(xt, yt), batch_size=batch_size,
                      shuffle=shuffle, drop_last=False)


def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            _, idx = torch.max(model(xb), 1)
            preds.append(idx.cpu().numpy())
            trues.append(yb.numpy())
    preds, trues = np.concatenate(preds), np.concatenate(trues)
    acc, p, r, f1 = get_clf_eval(trues, preds)
    cm = confusion_matrix(trues, preds, labels=list(range(6)), normalize="true")
    return acc, p, r, f1, cm


def main():
    set_seed()
    X, Y = load_antwerp_dataset()
    print(f"Antwerp 公開資料: X={X.shape}, Y={Y.shape}")
    print("SF 標籤分布(真實):", np.bincount(Y, minlength=6), "\n")

    model = CNNLSTM().to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"已載入 checkpoint: {CKPT_PATH}\n")

    # ---- 方案 A: Zero-shot ----
    loader = make_loader(X, Y, batch_size=512, shuffle=False)
    acc, p, r, f1, cm = evaluate(model, loader)
    print("=== [方案 A] Zero-shot 跨資料集 (NS-3 模擬 -> Antwerp 真實) ===")
    print(f"Acc={acc:.4f}  Precision={p:.4f}  Recall={r:.4f}  F1={f1:.4f}")
    print("每類 recall:", per_class_recall(cm))
    print("Confusion Matrix:\n", np.round(cm, 3), "\n")

    # ---- 方案 B: 凍結 CNN + 類別加權微調 ----
    x_tr, x_te, y_tr, y_te = train_test_split(
        X, Y, test_size=0.3, random_state=SEED, stratify=Y)
    tr_loader = make_loader(x_tr, y_tr, batch_size=512, shuffle=True)
    te_loader = make_loader(x_te, y_te, batch_size=512, shuffle=False)

    for name, param in model.named_parameters():
        if name.startswith("conv1d") or name.startswith("batch_norm"):
            param.requires_grad = False
    print("已凍結 CNN 特徵層, 僅微調 LSTM+FC")

    # 【關鍵】類別權重: inverse frequency, 放大少數類的錯誤懲罰
    class_counts = np.bincount(y_tr, minlength=6).astype(np.float32)
    weights = 1.0 / (class_counts + 1.0)
    weights = weights / weights.sum() * 6.0            # 正規化, 避免 loss 尺度暴衝
    weights_t = torch.tensor(weights, dtype=torch.float32).to(device)
    print("類別權重(SF7..SF12):", np.round(weights, 3))

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=1e-4, weight_decay=1e-6)
    loss_fn = nn.CrossEntropyLoss(weight=weights_t)    # <- 加權 loss

    for ep in range(1, 51):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()
        if ep % 10 == 0:
            acc, _, _, f1, _ = evaluate(model, te_loader)
            print(f"[Fine-tune] epoch {ep}  test Acc={acc:.4f}  F1={f1:.4f}")

    acc, p, r, f1, cm = evaluate(model, te_loader)
    print("\n=== [方案 B] 凍結 CNN + 類別加權微調後 ===")
    print(f"Acc={acc:.4f}  Precision={p:.4f}  Recall={r:.4f}  F1={f1:.4f}")
    print("每類 recall:", per_class_recall(cm))
    print("Confusion Matrix:\n", np.round(cm, 3))
    print("\n[解讀] 對比基本版(無權重 Acc=0.739/F1=0.628 但全猜SF7):")
    print("  若本版 Acc 下降但 F1 上升、對角線浮現 => 模型真正學會區辨少數類。")


if __name__ == "__main__":
    main()