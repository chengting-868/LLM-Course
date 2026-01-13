配置conda镜像源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
创建conda 环境
conda create -n CogReader python=3.10
激活conda环境
conda activate CogReader
安装依赖包
pip install -r require.txt
安装pytorch
pip3 install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
注意查看自己的cuda版本号与cu118号匹配，要求cuda版本号大于该版本号；

也可以直接下载cpu版本的pytorch

pip install torch==2.5.1 torchvision==0.20.1
使用cpu版本的pytorch需要将代码中的设备全部移到cpu上，及将代码中所有的cuda全部替换为cpu：

ctrl+f cuda --> cpu

另外有两个文件太大不方便上传，这里提供网盘链接：通过网盘分享的文件：Model等2个文件
链接: https://pan.baidu.com/s/1WJ-BPtmT2ZbHl646D7MsBw?pwd=ywfe 提取码: ywfe
其中pytorch_model.bin放在bert-base-uncased文件夹中，Model文件夹在根目录
