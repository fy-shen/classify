# 前言

现有的视频分类相关代码比较老旧，依赖过去的代码和早期版本的库，不利于模型修改和转换。

本项目参考 Ultralytics YOLO 代码逻辑，使用 yaml 配置文件来管理各种运行参数和模型构建，并使用 PyTorch 中的 DDP 模式来支持分布式训练。

# 安装

```bash
conda create -n torch2.7.0 python=3.9 -y
conda activate torch2.7.0
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

# 参考示例

（1）[ResNet+cifar10](configs/examples/res_cifar10.yaml)

```bash
python run.py --cfg configs/examples/res_cifar10.yaml
```

配置文件支持使用 PyTorch API 名称构建模型、数据、优化器等（大小写不敏感）

 - [torch.optim](https://docs.pytorch.org/docs/2.7/optim.html)
 - [torchvision.datasets](https://docs.pytorch.org/vision/0.22/datasets.html)
 - [torchvision.models](https://docs.pytorch.org/vision/0.22/models.html)

（2）[MLP+cifar10](configs/examples/mlp_cifar10.yaml)

使用自定义网络时，通过 `model_cfg` 指定模型的配置文件（[MLP配置文件](configs/models/MLP.yaml)），其书写方式与 Ultralytics 类似。
这里的模型名称 `model: MLP` 是因为在代码中注册了一个自定义的模型类 MLP，
通过这个注册机制，可以很方便的使用类名（大小写不敏感）来使用自定义的模型、数据集、损失、模块、数据处理等。

根据[注册代码](archs/__init__.py)，在构建自定义类时，可以通过装饰器自动注册，例如：

```python
from archs import register

@register('model')
class MLP(BaseModel):

@register('model')
class TSN(nn.Module):

@register('module')
class Mean(nn.Module):
```

