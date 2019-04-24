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
    "loss": "mse",#參考(keras.io/losses/)
    "optimizer": "adam",#參考(keras.io/optimizers/)
    "plan": "cnn",#可選cnn or lstm
    "save_model_dir": "saved_models",#訓練後的權重儲存資料夾
    "save_graph_dir": "graph"#圖表儲存資料夾
  },
  "training": {
    "batch_size": 39,#批次大小
    "epochs": 1000,#訓練次數
    "name": "tech_project"#專案名稱
  }
}

```

------------------

## 建立自己的模型
修改core/model.py檔案中的build_customized_model()
- 參考[https://keras.io/layers/core/#dense](https://keras.io/layers/core/#dense)建立模型


## 後續應用
- 把keras轉換成tensorflow檔案
[https://github.com/amir-abdi/keras_to_tensorflow](https://github.com/amir-abdi/keras_to_tensorflow)
- 把pb檔案放入android專案(修改路徑app/src/main/assets/tensorflow_inception_graph.pb)
[https://github.com/MindorksOpenSource/AndroidTensorFlowMachineLearningExample](https://github.com/MindorksOpenSource/AndroidTensorFlowMachineLearningExample)