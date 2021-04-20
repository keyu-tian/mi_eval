# 下游 MI 测试


### Dependencies

- 在 r0.3.3 环境下测试没有问题
- 少量依赖包列在 `requirements.txt` 中了


### Run

- 1. cd 到 `exp` 文件夹下的某个数据集名文件夹下。共四个数据集可测：活体 `liveness`，属性年龄 `age`，属性性别 `gender`，ImageNet 子集 `subimagenet`
- 2. 在 `cfg.yaml` 里面的 `checkpoints` 修改想测的 ckpt 的路径
- 3. 在 `cfg.yaml` 同级下 `sh ./out.sh` 或者 `sh ./spring.sh`（分别是 out 和 in 模式）
- 4. 大概 10 min 内可以计算完成，最终输出表格的 hy 是 MI(h, y)
- 5. 如果想计算 MI(h, x)，需要在 `cfg.yaml` 打开 `calc_hx`；因为目前 MI(h, x) 和下游 finetune 以后的相关性不强而且计算比较慢，所以都关掉 MI(h, x) 的计算了

### 结果打印示例

```txt
                          ckpt hy_mea hy_max hy_top hx_mea hx_max hx_top
0  DY_MTL_LV1_10_R50_convertBB   1.46    6.8    3.6     -1     -1     -1
1  DY_MTL_LV1_30_R50_convertBB      2   8.23   5.07     -1     -1     -1
2  DF_MTL_LV1_39_R50_convertBB   1.23   8.09   3.31     -1     -1     -1
```

（请主要参考 top 指标）
