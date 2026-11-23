# pairwise-sss-skin-thickness

## 领域

游戏、图形与多媒体

## 主题

皮肤次表面散射

## 说明

厚度随形变、LUT 曲率与核重算。

初始状态下 `graphics/sss_skin_thickness.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
