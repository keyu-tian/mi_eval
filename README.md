# 下游 MI 测试


### Dependencies

- 在 r0.3.3 环境下测试没有问题
- 少量依赖包列在 `requirements.txt` 中了


### Run

- 1. cd 到 `exp` 文件夹下的某个数据集名文件夹下，比如 `liveness_valset`。共四个数据集可测：活体 `liveness`，属性年龄 `age`，属性性别 `gender`，ImageNet 子集 `subimagenet`
- 2. 进入某个数据集文件夹后，选好上级路径想用的 `cfg.yaml`，比如 `../cfg_0426_siyu_with_liveness_head.yaml`，然后 `sh ./out.sh ../cfg_0426_siyu_with_liveness_head.yaml` 即可
- 3. 大概 10 min 内可以计算完成，最终输出表格中的 hy 的意思是 MI(h, y)

### 结果打印示例

```txt
                          ckpt hy_mea hy_max hy_top hx_mea hx_max hx_top
0  DY_MTL_LV1_10_R50_convertBB   1.46    6.8    3.6     -1     -1     -1
1  DY_MTL_LV1_30_R50_convertBB      2   8.23   5.07     -1     -1     -1
2  DF_MTL_LV1_39_R50_convertBB   1.23   8.09   3.31     -1     -1     -1
```

（请主要参考 top 指标）

- 结果也会以 `json` 格式，将 `pandas.DataFrame` 保存在当前文件夹内，例如 `results_gender_neib12_04-19_21-44-37.json`。
- 使用 `pandas.read_json(filename)` 可以加载这个 `DataFrame`。
