

## 环境配置：

1. 下载[Pychram](https://www.jetbrains.com/pycharm/)
2. 下载[Anaconda](https://www.anaconda.com)

参考环境搭配教程：[PyCharm+Conda](https://blog.csdn.net/weixin_45242930/article/details/135356097)

需要注册Conda账号，很简单。输入账号和密码即可。后续在邮箱中点击Verify email即可。

3. 下载完成后，打开已有

4. 配置conda镜像源

```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
```

5. 创建conda 环境

```bash
conda create -n CogReader python=3.10
```

6. 激活conda环境

```bash
conda activate CogReader
```

7. 安装依赖包

```bash
pip install -r require.txt
```

8. 安装pytorch

```bash
pip3 install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
```

注意查看自己的cuda版本号与**cu118**号匹配，要求cuda版本号大于该版本号；

也可以直接下载cpu版本的pytorch

```bash
pip install torch==2.5.1 torchvision==0.20.1
```

使用cpu版本的pytorch需要将代码中的设备全部移到cpu上，及将代码中所有的cuda全部替换为cpu：

```
ctrl+f cuda --> cpu
```

