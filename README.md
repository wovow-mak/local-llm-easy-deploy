# 本地大模型一键部署（Windows / 家用显卡）

> 一条命令在 Windows 上跑起本地大模型：自动检测显存 → 推荐配置 → 环境体检 → 开始对话。
> 不用懂原理，不用买服务器，有张显卡就行。

![Python](https://img.shields.io/badge/Python-3.10--3.12-blue)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 这是什么

一个面向新手的本地大模型部署脚本。核心是 `deploy.py`——它替你做完了最容易出错的三件事：

1. **环境体检**：Python 版本、显卡显存、推理库、模型文件，一次性检查完，缺什么直接告诉你补什么
2. **自动推荐**：读你的显存，告诉你该跑多大的模型、上下文开多少
3. **人话报错**：缺 DLL、缺模型、上下文超限，都给出具体命令，而不是甩一堆 traceback

我用的是 **RTX 4060 8G + Qwen2.5-1.5B**，实测 **105 tok/s**——比人打字快几十倍。

## 效果

<!-- 运行效果：体检 + 对话 -->
![运行效果：环境体检 + 对话](docs/screenshots/check.png)

```
===== 环境体检 =====
  [OK]   Python 3.12.8  (C:\Python312\python.exe)
  [OK]   显卡显存 8188 MiB（约 7 GB）
         推荐：Qwen2.5-7B Q4 / 1.5B Q4，上下文 8192
  [OK]   llama-cpp-python 0.2.90
  [OK]   CUDA 运行库目录 2 个（已自动注入 DLL 搜索路径）
  [OK]   GPU 加速可用
  [OK]   模型文件 qwen2.5-1.5b-instruct-q4_k_m.gguf（1066 MB）
====================
```

## 快速开始（3 步）

### 1. 装依赖

**强烈建议用虚拟环境**（不污染系统 Python，出问题删掉 `.venv` 重来即可）：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> ⚠️ **别用 `pip install llama-cpp-python` 裸装**——PyPI 上这个包只发布源码包（sdist），
> pip 会尝试本地编译，需要 CMake + MSVC 构建工具，新手几乎必失败。
> `requirements.txt` 里已经指定了官方预编译源，用 `-r` 装才会直接拿到 wheel。
>
> 国内网络慢的话，给 pip 加清华镜像（预编译 wheel 源不支持镜像加速，慢就只能等）：
> `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

**关于 `llama.dll` 缺依赖的问题**：`import llama_cpp` 偶尔会报
`Could not find module 'llama.dll' (or one of its dependencies)`。
原因是它依赖 CUDA 运行库（`cudart64_12.dll` 等），而这些 DLL 装在
`site-packages/nvidia/*/bin` 下，不在系统 PATH 里。

**本项目的 `deploy.py` 会在启动时自动把这两个目录注入 DLL 搜索路径**，你不用手工复制任何文件。
（手工复制 DLL 的做法在自己机器上能跑，换台机器就失效，所以没有采用。）

### 2. 下载模型（1GB，走国内镜像）

```bash
curl -L -o models/qwen2.5-1.5b-instruct-q4_k_m.gguf https://hf-mirror.com/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

> 国内直连 HuggingFace 很慢，`hf-mirror.com` 是官方镜像，把域名换掉即可。

### 3. 运行

```bash
python deploy.py
```

Windows 用户也可以直接双击 `deploy.bat`。

## 兼容性验证

同一个脚本在两套完全隔离的环境里都跑通了（这才叫"能给别人用"）：

| 环境 | 结果 |
|---|---|
| 系统 Python 3.12 + 全局安装 | ✅ 体检通过 · 105 tok/s |
| **全新 venv（`python -m venv .venv`，零手工配置）** | ✅ 体检通过 · 无需复制任何 DLL |

## 参数

| 参数 | 说明 | 示例 |
|---|---|---|
| 无 | 体检 → 加载 → 对话 | `python deploy.py` |
| `--check` | 只体检，不加载模型 | `python deploy.py --check` |
| `--ctx N` | 上下文长度，越大记得越多、越吃显存 | `--ctx 8192` |
| `--cpu` | 强制用 CPU（慢 5 倍，用于对比） | `--cpu` |
| `--gpu-layers N` | 放进显卡的层数，`-1` 为全部 | `--gpu-layers 20` |
| `--model 路径` | 指定模型文件 | `--model D:/models/qwen7b.gguf` |
| `--help` | 查看全部参数 | `--help` |

对话中可用：`q` 退出，`/clear` 清空对话记忆。

## 显存怎么选模型

| 显存 | 建议模型（4bit 量化） | 上下文 |
|---|---|---|
| 4G | Qwen2.5-1.5B | 4096 |
| **8G** | 1.5B 流畅 / 7B 勉强 | 8192 |
| 12-16G | 7B / 14B | 8192 |
| 24G+ | 32B | 8192 |

## 性能实测（RTX 4060 8G / Qwen2.5-1.5B Q4_K_M）

**GPU 层数对速度的影响**

| 配置 | 速度 |
|---|---|
| `n_gpu_layers=0`（纯 CPU） | 18.7 tok/s |
| `n_gpu_layers=-1`（全 GPU） | 105.7 tok/s |

**上下文窗口对显存的影响**

| n_ctx | 显存占用 |
|---|---|
| 未加载 | 596 MiB |
| 256 | 1798 MiB |
| 2048 | 2017 MiB |
| 8192 | 2185 MiB |
| 32768 | 3398 MiB |

> 结论：8G 显卡跑 1.5B 模型开 32K 上下文完全够，还剩 4.7G。
> 实测 KV cache 边际成本约 46 KB/token。

## 常见问题

<details>
<summary><b>1. 下载 447MB 的包要 2 小时</b></summary>

直连 GitHub 的速度只有 53.6 kB/s。用镜像手动下载，再本地安装：

```bash
curl -L -o llama_cpp.whl https://ghfast.top/https://github.com/abetlen/llama-cpp-python/releases/download/v0.2.90-cu121/llama_cpp_python-0.2.90-cp312-cp312-win_amd64.whl
pip install llama_cpp.whl
```
9 分钟搞定。
</details>

<details>
<summary><b>2. 装包报 <code>is not a supported wheel on this platform</code></b></summary>

Python 版本和 wheel 标签不一致。`cp312` 表示给 Python 3.12 的包。
确认当前解释器版本：`python -c "import sys; print(sys.version)"`，然后下对应版本的包。
</details>

<details>
<summary><b>3. import 报 <code>Could not find module 'llama.dll' (or one of its dependencies)</code></b></summary>

**显卡驱动 ≠ CUDA 运行库**。装驱动只能让系统识别显卡，程序要用 GPU 还得有 cudart/cublas。

```bash
pip install nvidia-cuda-runtime-cu12==12.1.105 nvidia-cublas-cu12==12.1.3.1
```

装完**直接重新运行 `python deploy.py` 就行**——脚本会自动把 `site-packages/nvidia/*/bin`
注入 DLL 搜索路径，不需要手工复制文件。

> 为什么不用"复制 DLL 到 `llama_cpp/lib`"这个网上常见的办法：它在你自己机器上有效，
> 但换个虚拟环境或换台电脑就又坏了，而且 pip 升级包之后会被清掉。
> 代码自适应比手工修补更可靠。
</details>

<details>
<summary><b>4. .bat 里的中文乱码，还报 "is not recognized as an internal or external command"</b></summary>

批处理存成了 UTF-8，cmd 按 GBK 解析。另存为 ANSI，或在开头加 `chcp 65001`。
</details>

<details>
<summary><b>5. 双击 .py 直接闪退</b></summary>

双击用的是系统默认 Python，可能没装依赖，报错还没看清窗口就关了。
**在终端里运行**，或用本项目提供的 `deploy.bat`（末尾有 pause，报错会停住）。
</details>

<details>
<summary><b>6. 换个目录运行就报 <code>Model path does not exist</code></b></summary>

相对路径的锚点是"你敲命令时所在的目录"，不是脚本所在位置。
本项目的 `deploy.py` 已用 `os.path.dirname(os.path.abspath(__file__))` 修正，从任何目录运行都可以。
自己写脚本时也照这个写法。
</details>

<details>
<summary><b>7. 参数没改，怎么突然变慢了 5 倍</b></summary>

先看 `config.json`。命令行传的参数会被记住，下次自动生效——如果之前试过 `--cpu`，
之后每次启动可能都在跑 CPU 模式。

v0.9 之后已修：**`--cpu` 这类临时开关不再写入配置**，只有 `--ctx` / `--gpu-layers` / `--model`
这种"你想长期改变默认值"的参数才会被记住，并且会打印 `已记住设置：...` 告诉你。

排查口诀：**程序变慢 → 先看配置文件里存了什么，再看命令行传了什么。**
</details>

## 目录结构

```
local-llm-easy-deploy/
├── deploy.py            # 主程序
├── deploy.bat           # Windows 双击启动器（支持传参）
├── requirements.txt
├── config.example.json  # 配置示例，运行后自动生成 config.json
├── models/              # 放 .gguf 模型文件（不提交到 git）
└── docs/screenshots/    # README 用的截图
```

## 更新日志

- **v0.9.1** — 自动注入 CUDA 运行库路径（不再需要手工复制 DLL）；`--cpu` 等临时开关不再写入配置；`--check` 增加 GPU 加速可用性检测；`requirements.txt` 指定预编译源，避免误触发源码编译
- **v0.9** — 首个版本：环境体检、显存推荐、友好报错、配置持久化

## 后续计划

- [ ] 支持自动下载模型（一条命令从零到对话）
- [ ] 加 Web 界面
- [ ] 挂载本地知识库（RAG）
- [ ] 推理加速对比（vLLM / TensorRT-LLM）

## 参考

- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [Qwen2.5 模型](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)

---

如果这个项目帮你省了时间，点个 **Star** ⭐ 就是对我最大的鼓励。
有问题欢迎提 Issue，看到都会回。

## License

MIT
