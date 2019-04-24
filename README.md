# KerasCustomized 客製化的keras模型

## 包含功能.
-包含簡單的卷積神經網路(CNN)跟長短期記憶卷積神經網路模型(LSTM)
-產生loss圖表
-產生訓練紀錄檔
-產生h5訓練後檔案

## 開始:準備訓練與測試資料
準備對應的numpy型態x_train, y_train訓練資料
(預設使用mnist資料集)

- x_train:(60000, 28, 28)#(資料數,資料欄,資料列)

- y_train:(60000,)#(結果數,)

並記得修改data.input_dim(資料欄)與data.input_timesteps(資料列)

------------------

## 依照資料以及需求調整參數
修改config.json檔案
```json

{
  "data": {
    "input_dim": 28,#資料欄數，決定輸入層
    "input_timesteps": 28,#資料列數，決定輸入層
    "num_classes": 10#分類數，決定輸出層
  },
  "model": {
    "loss": "mse",#參考 -[losses] (keras.io/losses/)
    "optimizer": "adam",
    "plan": "cnn",
    "save_dir": "saved_models"
  },
  "training": {
    "batch_size": 39,
    "epochs": 10,
    "name": "tech_project"
  }
}


```

