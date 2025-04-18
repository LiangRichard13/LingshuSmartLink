
# 灵枢智联

<div align="center">
  <img src="https://pic.imgdd.cc/item/6801e658218de299cab1fa8f.png" width="200" />
</div>

## 📝目录

- [📖 简介](#简介)
- [📊数据来源](#数据来源)
- [🔍RAG模块构建](#RAG模块构建)
- [😈多Agent架构](#多Agent架构)
- [🛠️ 使用方法](#使用方法)

## 📖 简介 <a id="简介"></a>

灵枢智联是一款基于多Agent协同架构的多模态中医智能诊疗系统，基于[LangChain](https://github.com/langchain-ai/langchain)框架进行搭建

系统通过深度融合舌象病理智能分析、动态问诊交互、中医典籍RAG检索、任务规划与迭代优化以及智能网络检索等核心模块，构建了从问诊到辨证的完整智能诊疗闭环。

系统最终生成的智能诊断报告包含以下专业内容：

- 患者基本信息
- 舌象描述
  - 舌苔特征
  - 可能的健康问题
- 诊断与辩证结论
  - 主诉症状归纳
  - 中医辩证要点
  - 诊断结论
  - 诊断依据
- 基础调整方案
  - 饮食调整方案
  - 饮食禁忌
  - 生活行为调整
  - 心理调整
- 中医治疗方案
  - 中药处方
  - 针灸方案
  - 推拿方案
  - 食疗推荐
- 预警方案
  - 红色预警
  - 橙色预警
  - 黄色预警

<div align="center">
  <img src="https://pic.imgdd.cc/item/6801fbda218de299cab4620a.png" width="100%" />
</div>


## 📊数据来源<a id="数据来源"></a>

舌苔病理分类训练数据来源：[项目首页 - 舌苔数据集-中医图像识别资源](https://gitcode.com/open-source-toolkit/7542e)共包含了2460张高质量舌苔图像，共包含以下六类：黑苔、地图舌苔、紫苔、红苔黄厚腻苔、红舌厚腻苔和白厚腻苔

RAG构建中医古籍资料来源：[中医药古籍文本](https://github.com/xiaopangxia/TCM-Ancient-Books)，共包含700篇中医古籍文本

## 🔍RAG模块构建<a id="RAG模块构建"></a>

使用大模型对收集到的中医药古籍文本中的81篇进行现代汉语翻译翻译,对翻译后的文档按照目录结构进行分块,使用[GTE文本向量-中文-通用领域-base](https://www.modelscope.cn/models/iic/nlp_gte_sentence-embedding_chinese-base/)进行向量嵌入并存储到Chroma向量数据库中

## 😈多Agent架构 <a id="多Agent架构"></a>

### 舌苔诊断Agent

#### 舌苔病理分类网络训练

1. 通过迁移学习方法使用预训练的ResNet18模型对舌苔图像进行分类，修改全连接层，使其输出类别数为6
2. 使用2460张经过数据增强（随机旋转、翻转、剪裁、缩放和颜色抖动）70%用于训练，15%用于验证，15%用于测试
3. 最后测试结果：Test Accuracy 98.65%

![](https://pic.imgdd.cc/item/6801f03f218de299cab2ac9b.png)

#### 舌苔诊断Agent构建

- 使用[阶跃星辰step-1o-turbo-vision](https://platform.stepfun.com/)视觉大模型并结合训练好的ResNet18舌苔图像分类网络进行舌苔病理诊断并输出以诊断信息,包含舌苔特征和可能的健康问题

### 信息收集Agent

- 使用[DeepSeek-V3](https://platform.deepseek.com/)采用多轮对话的形式引导用户回答个人基本信息,症状及持续时间,既往史及家族史等对于诊断有用的信息,最后输出患者的信息摘要

### 任务规划Agent

- 使用[DeepSeek-V3](https://platform.deepseek.com/),结合给出的任务大致框架和患者的背景信息规划出问诊的具体步骤并输出任务列表

### 行动Agent

- 使用[DeepSeek-V3](https://platform.deepseek.com/),根据任务内容结合RAG检索到的参考资料和患者的信息执行任务，输出任务结果，并根据指导Agent的意见进行反复修改

### 指导Agent

- 使用[DeepSeek-V3](https://platform.deepseek.com/),根据任务内容结合患者信息指导行动Agent,给出具体的修改建议并决定是否执行下一个任务

### 总结Agent

- 使用[DeepSeek-V3](https://platform.deepseek.com/),负责总结行动Agent和指导Agent每次任务的多轮对话中的有效信息并输出该此任务的完整解决方案

### 报告生成Agent

- 使用[DeepSeek-V3](https://platform.deepseek.com/),首先会提取出报告中用户难以理解的中医术语、药材名称、穴位名称，并使用[百度词条API](https://baike.deno.dev/)和[SerperAPI](https://serper.dev/)进行网络搜索

- 参考搜索到的结果和任务解决方案分阶段生成markdown格式的报告并在markdown文档中加入必要的解释以及相关网络链接、图片和相关养生文章


## 🛠️ 使用方法 <a id="使用方法"></a>

* 下载项目并添加路径

```shell
git clone <GithubRepo>
```

* 安装Conda环境

```shell
conda create -n LinShuSmartLink python=3.12
conda activate LinShuSmartLink
pip install -r requirements.txt
```

* 准备APIKEY

在`<项目路径>/backend`下新建`.env`文件,并按照以下格式输入自己的APIKEY
```shell
SERPER_API_KEY=<YOUR_SERPER_API_KEY> # 设置SerperAPI的apikey用于提供网络搜索工具

DEEP_SEEK_API_KEY=<YOUR_DEEP_SEEK_API_KEY> # 用于诊断任务执行

STEP_FUN_API_KEY=<YOUR_STEP_FUN_API_KEY> # 用于舌苔病理诊断
```

* 训练分类模型

运行训练脚本
```shell
python <项目路径>/backend/model/tongue_coating_classify/train/train.py
```

将产生的权重文件`tongue_coating_resnet18.pth`保存至`<项目路径>/backend/model/tongue_coating_classify`下

- 下载Embedding Model
```shell
modelscope download --model iic/nlp_gte_sentence-embedding_chinese-base --cache_dir <项目路径>/model/embedding_model
```

- 运行向量数据库构建脚本
```shell
python <项目路径>/workspace/RAG/processed_rag_data.py
```
- 添加需要诊断的舌苔图片到`<项目路径>/backend/static/images`中
- 🚀启动项目
```shell
python <项目路径>/backend/agentsRunner.py
```

输入舌苔图片路径:`<项目路径>/backend/static/images/<你的舌苔图片文件>`

## 🎥 演示DEMO📕报告样例📑交互记录 <a id="演示DEMO"></a>

- [演示视频](https://github.com/LiangRichard13/LingshuSmartLink/blob/master/example/%E7%81%B5%E6%9E%A2%E6%99%BA%E8%81%94%E6%BC%94%E7%A4%BA%E8%A7%86%E9%A2%91.mp4)

- [报告示例](https://github.com/LiangRichard13/LingshuSmartLink/blob/master/example/markdown_20250418_155229.md)

- [交互记录](https://github.com/LiangRichard13/LingshuSmartLink/blob/master/example/%E4%BA%A4%E4%BA%92%E8%AE%B0%E5%BD%95.txt)
