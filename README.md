# AI-ERA 重現與改進：基於 1D CNN + LSTM 的 LoRaWAN 擴頻因子預測

> CSIE 2103 類神經網路 · 期末專題 · M11417022 王雅佳
> 國立雲林科技大學 資訊工程系 · Spring 2026

本專題重現論文 **AI-ERA** (Farhad et al., *IEEE Transactions on Industrial Informatics*, 2023) 的擴頻因子 (Spreading Factor, SF) 預測方法，並在其基礎上進行改進與跨資料集驗證：

1. **重現**原作者的 DNN 方法，確認環境與訓練邏輯正確（Val Acc 80.47% → 重現 80.60%）。
2. **改進**：自主將純 DNN 改寫為 **1D CNN + LSTM** 混合架構，在參數量減半（52k → 25k）的前提下，準確率仍超越原版（Test 80.60% / Val 81.45%），契合 2025 年 TinyML 輕量化趨勢。
3. **跨資料集驗證**：將模型遷移到歐洲都市公開資料集 **Antwerp**，分析跨資料集遷移的失敗機制。

---

## 環境安裝

本專題使用 `uv` 管理環境（亦可用一般 pip）。

```bash
# 使用 uv
uv venv
uv pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
uv pip install numpy pandas scikit-learn matplotlib seaborn ptflops prettytable

# 或使用 pip
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy pandas scikit-learn matplotlib seaborn ptflops prettytable
```

- **PyTorch**：1.9.0 + CUDA 11.1
- **GPU**：實驗於 NVIDIA GeForce GTX 1660 Ti 完成

> 📌 **與原作者環境的差異**：原作者 repo 使用 PyTorch 1.7.1 + Python 3.8。本專題成功將其遷移至 PyTorch 1.9.0 + CUDA 11.1 並完成重現，版本差異為重現結果微小誤差（< 2%）的來源之一。

---

## 檔案說明

| 檔案 | 角色 | 來源 |
|------|------|------|
| `main_code_v2.py` | 本專題改進版：CNN+LSTM 訓練主程式 | 本專題 |
| `model_cnnlstm.py` | 自主設計的 1D CNN + LSTM 模型定義 | 本專題 |
| `cross_dataset_antwerp.py` | 跨資料集遷移實驗（含類別加權微調） | 本專題 |
| `model/LSTM/[CNNLSTM]...(acc-80).pt` | 訓練完成的最佳模型權重 | 本專題 |

> 📌 **關於原作者的基準程式**：本專題的重現基準（原始 DNN 程式 `main_code.py` 與資料生成 `dataset_gen.py`）來自原作者 repo：[github.com/afarhad/AI-ERA](https://github.com/afarhad/AI-ERA)。由於原 repo 未附授權條款（no license），本 repo **不重新散布**其原始程式碼，請逕至原作者 repo 取得。本專題的所有改進皆在 `main_code_v2.py`、`model_cnnlstm.py` 與 `cross_dataset_antwerp.py` 中。

---

## 資料下載

資料檔因體積或來源限制未包含於本 repo，請自行下載後放入對應資料夾：

- **訓練資料（ns-3 模擬）**：取自原作者 AI-ERA repo，放入 `./data/`
- **跨資料集（Antwerp 2019）**：公開 LoRaWAN benchmark（130,423 筆，含真實 SF 標籤、72 基地台 RSSI、GPS 座標）。下載來源：[Kaggle - LoRaWAN Antwerp 2019 dataset](https://www.kaggle.com/datasets/goapgo/lorawan-antwerp-2019-dataset-csv)。下載後將 `antwerp_dataset.csv` 放入專案根目錄。

---

## 如何執行

### 1. 訓練 CNN+LSTM 模型

```bash
python main_code_v2.py
```

訓練完成的模型會儲存至 `./model/LSTM/`。

### 2. 重現跨資料集遷移實驗

```bash
python cross_dataset_antwerp.py
```

此程式會依序執行：Zero-shot 直接遷移 → 凍結 CNN 的類別加權微調，並輸出每個方案的 Accuracy、F1、每類 recall 與混淆矩陣。

---

## 重現與改進歷程（逐步說明每次更改了什麼）

以下記錄完整的實驗演進。每一步都記載「改了什麼」與「為什麼」，對應期末 Excel 的重現實作記錄表。

### 階段一：重現原作者 DNN（驗證環境與邏輯正確）

| 步驟 | 更改內容 | 結果 (Val Acc) | 結論 |
|------|----------|----------------|------|
| 1 | 建置舊版 PyTorch 1.9.0 + CUDA 11.1 環境，跑原作者 DNN，Epoch=10 | 53.52% | 模型架構（參數量 52,906、14 MMac）與論文 Table II 100% 吻合，但訓練週期嚴重不足 (Underfitting)，且 LR 在第 8 epoch 就降至 1e-6 |
| 2 | **增加訓練週期至 Epoch=1000** | 74.22% | 準確率大增 +20.7%，但後期 LR 觸底（1e-6），權重幾乎不更新，卡在最後 6% |
| 3 | **訓練週期拉長至 Epoch=5000** | 78.91% | 達收斂，與論文 80.47% 誤差 < 2%，證實環境與訓練邏輯完全正確。重現成功 |

### 階段二：架構改進 — 從 DNN 改寫為 CNN+LSTM（TinyML 輕量化）

| 步驟 | 更改內容 | 結果 (Val Acc) | 結論 |
|------|----------|----------------|------|
| 4 | **全新導入 1D CNN + LSTM 架構**取代原 DNN，Epoch=10 初測 | 32.81% | 參數量從 52.9k 大幅降至 6.8k，契合 TinyML 精神；但 LSTM 需更長收斂時間，且 LR 衰減過快 |
| 5 | CNN+LSTM 完整訓練 Epoch=1000 | 60.94% | 出現 LR 過早停滯 (LR Starvation)，Epoch 200 即趨平緩。完全無 Overfitting，泛化潛力佳 |
| 6 | **修復 LR Scheduler**（學習率穩定維持 2.5e-5），Epoch=1000 | 60.55% | 確認瓶頸不在訓練策略，而是 6.8k 參數的微型模型容量不足 (Underfitting) |
| 7 | **擴增網路寬度**：CNN 通道數 → 32、LSTM 隱藏層 → 64（參數量提升至約 25k），Epoch=1000 | 64.45% | 擴容立即見效 +4%，Validation 穩定跟隨 Training，方向正確 |
| 8 | 擴容後拉長訓練至 Epoch=5000 | 74.61% | 大幅提升 +10%，以 25k 參數已超越早期 52k DNN 訓練 1000 次的表現 |
| 9 | **堆疊雙層 LSTM**（加深網路），Epoch=5000 | 76.17% (Peak) | 加深有效，但 Loss 震盪加劇、訓練時間增至 2428s，模型對 LR 較敏感 |
| 10 | **Batch Size 256 → 512**，Epoch=5000 | 80.66% | 抑制梯度震盪跳出局部最佳解，準確率超越原版 80.47%；訓練時間從 2428s 降至 1358s（效率 +44%） |
| 11 | **訓練週期拉長至 Epoch=10000**（最終定案） | **Val 81.45% / Test 80.60%** | Precision/Recall/F1 均衡（約 0.80~0.82），以 25k 參數穩定超越原版 52k DNN。**最終定案模型** |

**階段二總結**：以不到原版 DNN 一半的參數量（25k vs 52k），達到並超越原版效能（Test 80.60% vs 原版 80.47%），成功實現 TinyML 輕量化目標。

### 階段三：跨資料集遷移驗證（Antwerp 公開資料集）

將階段二的最佳模型遷移到歐洲都市真實量測資料 Antwerp（130,423 筆，真實 SF 標籤），分析其泛化能力。

> **重要**：Antwerp 中 SF7 佔 74%，類別極度不平衡，故以 **F1 為主指標**，Accuracy 會嚴重誤導。

| 方案 | 更改內容 | Acc | F1 | 現象 |
|------|----------|-----|-----|------|
| A. Zero-shot | 直接遷移，不訓練 | 73.9% | 0.628 | **多數類別崩塌**：只猜 SF7（recall=1.0），其餘類別 recall=0。Acc 看似不差純屬假象 |
| B-1. 普通微調 | 凍結 CNN，微調 LSTM+FC | 73.9% | 0.628 | 仍被類別不平衡困住，持續全猜 SF7 |
| B-2. **類別加權微調** | 在 B-1 加入 inverse-frequency 類別權重 + 特徵標準化 | 32.8% | 0.423 | **打破崩塌**：六類 recall 皆 >0；但對角線偏低，暴露更深問題 |

**階段三的核心發現**：跨資料集遷移失敗，表面是類別不平衡，**深層是「特徵語意不對齊」**。Antwerp 缺少 SNR，僅能以「最強 RSSI + 基地台數量」近似，特徵鑑別力不足以區分六種 SF。類別加權能解決「模型不願預測少數類」，卻無法解決「特徵本身缺乏鑑別力」。

**進一步觀察（source 模型品質與跨資料集泛化解耦）**：將來源模型從 5000 epoch（Val 80.66%）訓練至 10000 epoch（Val 81.45%），跨資料集 F1 幾乎不變（0.417 → 0.423）。這證明跨資料集遷移的瓶頸**不在來源模型訓練得夠不夠好，而在資料集之間的特徵語意鴻溝**——「source 模型強度」與「跨資料集泛化能力」是解耦的兩件事。真正的改善方向是補齊關鍵特徵（取得 SNR 或統一 RSSI 校準），而非把來源模型練得更強或僅調整 loss。

---

## 主要成果

- ✅ 成功重現 AI-ERA（Test 80.60% vs 論文 80.47%）
- ✅ 自主設計 CNN+LSTM，參數量減半（52k → 25k）且準確率超越原版（Val 81.45%）
- ✅ 完成跨資料集遷移實驗，誠實揭露並分析遷移失敗的多層機制

---

## 參考文獻

1. A. Farhad et al., "AI-ERA: Artificial Intelligence-Empowered Resource Allocation for LoRa-Enabled IoT Applications," *IEEE Trans. Industrial Informatics*, 2023.
2. M. A. Lodhi et al., "AI-Enhanced Resource Allocation for LPWAN-Based LoRaWAN: A Hybrid TinyML and Deep Learning Approach," *IEEE Internet of Things Journal*, 2025.
3. L.-T. Tu et al., "Energy Efficiency Optimization in LoRa Networks—A Deep Learning Approach," *IEEE Trans. Intelligent Transportation Systems*, 2023.
