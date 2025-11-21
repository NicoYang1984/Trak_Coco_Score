# Trak_Coco_Score

## 使用说明
```
# 使用所有8个GPU并行处理不同的窗口
python main_controller.py --start 0 --end 100 --gpus 8

# 使用4个GPU处理窗口
python main_controller.py --start 0 --end 50 --gpus 4

# 处理特定范围的窗口
python main_controller.py --start 10 --end 30 --gpus 2
```

## 使用前注意
- 修改`window_controller.py`和main_controller.py`当中的：
   - COCO_ROOT 变量（存放coco数据集的路径）
   - SAVE_ROOT 变量（存放打分结果的路径）
