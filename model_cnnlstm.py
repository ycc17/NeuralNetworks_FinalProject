import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        
        # 1. 擴充 CNN：將萃取的特徵通道從 16 提升到 32
        self.conv1d = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(32) 
        
        # 2. 擴充 LSTM：接收 32 維輸入，隱藏層記憶體從 32 擴張到 64
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=2, batch_first=True)
        
        # 3. 分類層：接收 64 維特徵，輸出 6 種 SF
        self.fc = nn.Linear(64, 6)

    def forward(self, x):
        # [步驟 A] 資料重塑 (Reshape)
        # 使用 -1 讓 PyTorch 自動推斷 Batch 的大小，
        # 這樣不管是實際訓練還是 torchsummary 測試，都不會報錯！
        x = x.view(-1, 6, 4)
        
        # [步驟 B] 適配 CNN 輸入格式
        # (Batch_size, 6, 4) -> (Batch_size, 4, 6)
        x = x.transpose(1, 2)
        
        # [步驟 C] 經過 CNN 萃取特徵
        x = self.conv1d(x)         
        x = self.batch_norm(x)
        x = self.relu(x)
        
        # [步驟 D] 適配 LSTM 輸入格式
        # (Batch_size, 16, 6) -> (Batch_size, 6, 16)
        x = x.transpose(1, 2)
        
        # [步驟 E] 經過 LSTM 處理時間序列
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # [步驟 F] 擷取最終特徵與分類
        last_time_step = lstm_out[:, -1, :] 
        
        out = self.fc(last_time_step)       
        return out

# 測試模型架構與張量維度
if __name__ == "__main__":
    # 模擬一筆 Batch Size 為 256，特徵為 24 的輸入
    dummy_input = torch.randn(256, 24)
    model = CNNLSTM()
    output = model(dummy_input)
    
    print(f"輸入維度: {dummy_input.shape}")
    print(f"輸出維度: {output.shape} (對應 6 種 SF 的 logits)")