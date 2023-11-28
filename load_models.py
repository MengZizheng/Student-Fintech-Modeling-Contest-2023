# 如何加载models中的模型
import lightgbm as lgb 
import numpy as np 


y_pred = []
for i in range(10):
    model = lgb.Booster(model_file=f'./models/{i}.txt')
    y_pred_ = model.predict(X_test)    
    y_pred.append(y_pred_)

y_pred = np.array(y_pred)
