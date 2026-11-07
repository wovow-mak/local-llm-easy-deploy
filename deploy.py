# -*- coding: utf-8 -*-
"""
deploy.py v1.0 —— 一键部署脚本（多模型版）

用法：
  python deploy.py                  # 用上次的模型直接开聊
  python deploy.py --list           # 看支持哪些模型、哪些已下载
  python deploy.py --model llama    # 换模型：qwen / llama / glm
  python deploy.py --check          # 只做环境体检，不启动
  python deploy.py --ctx 8192       # 手动指定上下文长度
  python deploy.py --cpu            # 强制用 CPU（没显卡或想对比速度时用）

v1.0 相比 v0.9 的变化：
  - 模型注册表（MODEL_REGISTRY）：加模型只加一条字典，其它代码不动
  - 统一接口：--model 支持别名 / 文件名 / 绝对路径三种写法
  - --list 一眼看出哪些模型已下载、没下载的给下载命令

设计原则（写给别人用的脚本，和写给自己用的脚本，区别就在这）：
  1. 路径以脚本自身为锚点，在哪个目录运行都能用
  2. 参数可配置，不写死在代码里
  3. 出错给人话，不甩一堆 traceback
  4. 先体检再运行，别等跑到一半才报错
  5. 差异集中到一张表里，别散落在代码各处
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import sysconfig
import time
import unicodedata
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
# 模型目录支持用环境变量覆盖（模型放在移动硬盘、或者想共用一份模型时很有用）
MODELS_DIR = os.environ.get("LLM_MODELS_DIR") or os.path.join(ROOT, "models")
CONFIG_PATH = os.path.join(ROOT, "config.json")

_cuda_dll_dirs = []


def prepare_cuda_dlls():
    """把 pip 装的 CUDA 运行库目录注入 DLL 搜索路径。

    背景：llama.dll 依赖 cudart64_12.dll / cublas64_12.dll / cublasLt64_12.dll。
    这几个 DLL 由 pip 包 nvidia-cuda-runtime-cu12、nvidia-cublas-cu12 提供，
    装在 site-packages/nvidia/*/bin 下——**不在系统 PATH 里**，
    所以 import llama_cpp 会报 "Could not find module ... (or one of its dependencies)"。

    错误示范：手工把 DLL 复制到 site-packages/llama_cpp/lib/。
             在自己机器上能跑，换台机器 / 换个虚拟环境就失效，别人拿你的项目还是跑不起来。
    正确做法：启动时把目录加进 PATH，代码自适应，在哪都能跑。
    """
    if _cuda_dll_dirs:
        return _cuda_dll_dirs
    paths = sysconfig.get_paths()
    found = []
    for key in ("purelib", "platlib"):
        root = paths.get(key)
        if root:
            found += glob.glob(os.path.join(root, "nvidia", "*", "bin"))
    found = sorted(set(found))
    if found:
        os.environ["PATH"] = os.pathsep.join(found) + os.pathsep + os.environ.get("PATH", "")
        _cuda_dll_dirs.extend(found)
    return found

# ---------------- 模型注册表 ----------------
# 一个模型 = 一个字典。新增模型只需要在这里加一条，其它代码一行都不用改。
# 这就是"统一接口"的意义：把差异集中到一个地方，而不是散落在代码各处。
#
# chat_format 说明：
#   None      → 用 GGUF 文件里自带的对话模板（现在多数模型都有，最省事）
#   "chatglm3" → 文件里没模板时手动指定格式，否则会退化成 llama-2 模板，回答会乱
MODEL_REGISTRY = [
    {
        "alias": "qwen",
        "name": "Qwen2.5-1.5B-Instruct",
        "family": "Qwen（阿里）",
        "file": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "size_mb": 1066,
        "url": "https://hf-mirror.com/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "url_alt": "https://modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/master/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "chat_format": None,
    },
    {
        "alias": "llama",
        "name": "Llama-3.2-1B-Instruct",
        "family": "Llama（Meta）",
        "file": "llama-3.2-1b-instruct-q4_k_m.gguf",
        "size_mb": 770,
        "url": "https://hf-mirror.com/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "url_alt": "https://modelscope.cn/models/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/master/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "chat_format": None,
    },
    {
        "alias": "internlm",
        "name": "InternLM2.5-1.8B-Chat",
        "family": "InternLM（上海 AI 实验室）",
        "file": "internlm2_5-1_8b-chat-q4_k_m.gguf",
        "size_mb": 1118,
        "url": "https://hf-mirror.com/Shanghai_AI_Laboratory/internlm2_5-1_8b-chat-gguf/resolve/main/internlm2_5-1_8b-chat-q4_k_m.gguf",
        "url_alt": "https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2_5-1_8b-chat-gguf/resolve/master/internlm2_5-1_8b-chat-q4_k_m.gguf",
        "chat_format": None,
    },
    {
        "alias": "glm",
        "name": "GLM-Edge-1.5B-Chat",
        "family": "GLM（智谱）",
        "file": "glm-edge-1.5b-chat-q4_k_m.gguf",
        "size_mb": 935,
        "url": "https://hf-mirror.com/zai-org/glm-edge-1.5b-chat-gguf/resolve/main/ggml-model-Q4_K_M.gguf",
        "url_alt": "https://modelscope.cn/models/ZhipuAI/glm-edge-1.5b-chat-gguf/resolve/master/ggml-model-Q4_K_M.gguf",
        "chat_format": None,
        # 2026-11-07 实测：zai-org 这份 GGUF 的 QKV 是合并张量，而 llama.cpp 的
        # chatglm 实现要求分离的 attn_q/attn_k/attn_v → 0.2.90 和 0.3.4 都加载失败，
        # 报错 missing tensor 'blk.0.attn_qkv.weight'。文件没坏，是张量布局对不上。
        "unsupported": "GGUF 张量布局与 llama.cpp 的 chatglm 实现不兼容",
    },
]

# 显存 -> (推荐模型, 推荐上下文) 的对照表
VRAM_TABLE = [
    (24000, "Qwen2.5-32B / 14B Q4", 8192),
    (12000, "Qwen2.5-14B / 7B Q4", 8192),
    (7000, "Qwen2.5-7B Q4 / 1.5B Q4", 8192),
    (4000, "Qwen2.5-1.5B Q4", 4096),
    (0, "Qwen2.5-0.5B Q4（很吃力）", 2048),
]

DEFAULT_CONFIG = {
    "model": "qwen",
    "n_ctx": 2048,
    "n_gpu_layers": -1,
    "n_threads": 0,
}


def _entry(m, path=None):
    """把注册表条目变成运行期用的模型对象（含"文件是否完整"的判断）"""
    e = dict(m)
    e["path"] = path or os.path.join(MODELS_DIR, m["file"])
    e["downloaded"] = False
    e["actual_mb"] = 0.0
    if os.path.exists(e["path"]):
        e["actual_mb"] = os.path.getsize(e["path"]) / 1024 / 1024
        expect = m.get("size_mb")
        if expect:
            # 允许 2% 误差；差太多说明是下载中途的文件（.gguf 不完整加载会直接报错）
            e["downloaded"] = abs(e["actual_mb"] - expect) <= max(5, expect * 0.02)
        else:
            e["downloaded"] = e["actual_mb"] > 1
    return e


def resolve_model(arg):
    """把 --model 的值解析成模型对象。三种写法都支持：

      --model qwen              # 别名（最常用）
      --model xxx.gguf          # models/ 目录下的文件名
      --model D:/models/x.gguf  # 任意路径
    """
    arg = (arg or DEFAULT_CONFIG["model"]).strip()
    low = arg.lower()

    for m in MODEL_REGISTRY:
        if low in (m["alias"], m["name"].lower(), m["file"].lower()):
            return _entry(m)

    if low.endswith(".gguf"):
        path = arg if os.path.isabs(arg) else os.path.join(MODELS_DIR, os.path.basename(arg))
        for m in MODEL_REGISTRY:
            if m["file"].lower() == os.path.basename(path).lower():
                return _entry(m, path)
        # 注册表里没有：当自定义模型处理，照样能跑
        return {"alias": "custom", "name": os.path.basename(path), "family": "自定义模型",
                "file": os.path.basename(path), "url": "", "chat_format": None,
                "path": path, "downloaded": os.path.exists(path)}

    print(f"不认识的模型：{arg}")
    print("可选别名：" + " / ".join(m["alias"] for m in MODEL_REGISTRY))
    sys.exit(1)


def _pad(s, width):
    """按终端显示宽度补空格：中文/全角字符占 2 列，str.ljust 只按字符数算，会错位"""
    shown = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)
    return s + " " * max(0, width - shown)


def list_models(current_alias=""):
    """列出所有支持的模型，以及本地有没有下载"""
    print("\n支持的模型：")
    for m in MODEL_REGISTRY:
        e = _entry(m)
        if e["downloaded"]:
            flag, size = "已下载", f"（{e['actual_mb']:.0f} MB）"
        elif e["actual_mb"] > 1:
            flag, size = "不完整", f"（{e['actual_mb']:.0f}/{m.get('size_mb', '?')} MB，可能正在下载）"
        else:
            flag, size = "未下载", ""
        mark = " ← 当前" if m["alias"] == current_alias else ""
        print(f"  {_pad(m['alias'], 10)}{_pad(m['name'], 28)}"
              f"{_pad(m['family'], 30)}[{flag}]{size}{mark}")
        if m.get("unsupported"):
            print(f"         [不可用] {m['unsupported']}")
            print("         这个模型的 GGUF 在当前引擎上加载不了，换一个别名就能跑")
            continue
        if not e["downloaded"]:
            print(f"         下载：python deploy.py --download {m['alias']}")
            if m.get("url_alt"):
                print(f"         主源慢时可换：{m['url_alt']}")
    print("\n用法：python deploy.py --model qwen | llama | internlm\n")


def _fetch(url, path, expect_mb=None):
    """断点续传下载，带进度。返回 True 表示成功。"""
    have = os.path.getsize(path) if os.path.exists(path) else 0
    req = urllib.request.Request(url, headers={"User-Agent": "deploy.py/1.0"})
    if have:
        req.add_header("Range", f"bytes={have}-")
        if have >= 1024 * 1024:
            print(f"    从 {have / 1024 / 1024:.1f} MB 处续传")
        else:
            print(f"    从 {have / 1024:.0f} KB 处续传")
    with urllib.request.urlopen(req, timeout=60) as r:
        # 判断是否真的续传成功，不能只看状态码：
        # ModelScope/OSS 这类服务会返回 200 但也带 Content-Range（标准要求 206），
        # 只看 status 会把"半截响应"当成"完整响应"，把已下好的部分截断。
        content_range = r.headers.get("Content-Range") or ""
        if have and not content_range:
            print("    服务器不支持续传，从头下载")
            have = 0
        mode = "ab" if have else "wb"
        total = 0
        if "/" in content_range:
            try:
                total = int(content_range.rsplit("/", 1)[1])
            except ValueError:
                total = 0
        if not total:
            total = int(r.headers.get("Content-Length") or 0) + have
        done = have
        t0 = last = time.time()
        with open(path, mode) as f:
            while True:
                chunk = r.read(256 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - last >= 0.5:
                    last = now
                    speed = (done - have) / max(now - t0, 1e-6) / 1024 / 1024
                    pct = done / total * 100 if total else 0
                    print(f"\r    {done / 1024 / 1024:6.0f} / {total / 1024 / 1024:.0f} MB"
                          f"  {pct:5.1f}%  {speed:4.1f} MB/s", end="", flush=True)
    print()
    if expect_mb:
        got = os.path.getsize(path) / 1024 / 1024
        if abs(got - expect_mb) > max(5, expect_mb * 0.02):
            print(f"    [失败] 文件大小不对：{got:.0f} MB，应为 {expect_mb} MB")
            return False
    return True


def download_model(alias):
    """一条命令从零到可对话：下模型 + 校验大小"""
    model = resolve_model(alias)
    if model["downloaded"]:
        print(f"{model['name']} 已经下载好了（{model['actual_mb']:.0f} MB），直接用就行")
        return True
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"下载 {model['name']}（约 {model.get('size_mb', '?')} MB）→ {model['path']}")
    for label, url in (("主源", model.get("url")), ("备用源", model.get("url_alt"))):
        if not url:
            continue
        print(f"  {label}：{url.split('/')[2]}")
        try:
            if _fetch(url, model["path"], model.get("size_mb")):
                print("  下载完成，校验通过")
                return True
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"  失败：{type(e).__name__}: {e}")
    print("两个源都失败了。可以稍后重试（会续传），或手动用浏览器下载后放到 models/ 目录。")
    return False



def ok(msg):
    print(f"  [OK]   {msg}")


def warn(msg):
    print(f"  [注意] {msg}")


def fail(msg, hint=""):
    print(f"  [失败] {msg}")
    if hint:
        print(f"         → {hint}")
    return False


# ---------------- 1. 环境体检 ----------------
def check_python():
    v = sys.version_info
    if v < (3, 9):
        return fail(f"Python 版本太旧：{v.major}.{v.minor}",
                    "建议用 Python 3.10~3.12")
    ok(f"Python {v.major}.{v.minor}.{v.micro}  ({sys.executable})")
    return True


def check_llama_cpp():
    dlls = prepare_cuda_dlls()
    try:
        import llama_cpp
        ok(f"llama-cpp-python {llama_cpp.__version__}")
        if dlls:
            ok(f"CUDA 运行库目录 {len(dlls)} 个（已自动注入 DLL 搜索路径）")
        try:
            if llama_cpp.llama_supports_gpu_offload():
                ok("GPU 加速可用")
            else:
                warn("这个构建不带 GPU 加速（纯 CPU 版），速度会慢 5 倍以上")
        except Exception:
            pass
        return True
    except ModuleNotFoundError:
        return fail(
            "没装 llama-cpp-python",
            "pip install llama-cpp-python==0.2.90 --prefer-binary "
            "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121",
        )
    except Exception as e:
        # 最常见：llama.dll 缺依赖（缺 CUDA 运行库）
        return fail(
            f"库装上了但加载失败：{type(e).__name__}",
            "缺 CUDA 运行库 → pip install nvidia-cuda-runtime-cu12==12.1.105 "
            "nvidia-cublas-cu12==12.1.3.1\n"
            "         装完重新运行本脚本即可（它会自动把 DLL 目录加进搜索路径）",
        )


def detect_vram():
    """返回显存 MiB；没有 N 卡返回 None"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        return int(out.stdout.strip().split("\n")[0])
    except Exception:
        return None


def check_gpu():
    vram = detect_vram()
    if vram is None:
        warn("没检测到 NVIDIA 显卡（可能是 A 卡/核显，或没装驱动）→ 只能用 CPU，速度会慢 5 倍以上")
        return None
    ok(f"显卡显存 {vram} MiB（约 {vram // 1024} GB）")
    for threshold, model, ctx in VRAM_TABLE:
        if vram >= threshold and threshold > 0:
            print(f"         推荐：{model}，上下文 {ctx}")
            break
    return vram


def check_model(model):
    path = model["path"]
    if model["downloaded"]:
        ok(f"模型 {model['name']}（{model['actual_mb']:.0f} MB）")
        return True
    if model.get("actual_mb", 0) > 1:
        return fail(
            f"模型文件不完整：{model['actual_mb']:.0f} MB / 应有 {model['size_mb']} MB",
            "下载没完成（或中断了）→ 用 curl -C - 续传，别直接运行",
        )
    if model.get("url"):
        hint = f"先下载：\n         curl -L -o models/{model['file']} {model['url']}"
    else:
        hint = "检查路径是否正确，或用 --list 看有哪些模型可用"
    return fail(f"模型不存在：{path}", hint)


def gguf_arch(path):
    """只读 GGUF 头部的第一个元数据（几乎总是 general.architecture），不加载模型。

    用途：模型加载失败时，先告诉用户"这是哪个架构的模型"，
    而不是让他对着 check_tensor_dims 这种报错猜。
    """
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            f.read(4)          # version (uint32)
            f.read(8)          # tensor_count (uint64)
            f.read(8)          # metadata_kv_count (uint64)
            klen = int.from_bytes(f.read(8), "little")
            key = f.read(klen).decode("utf-8", "ignore")
            vtype = int.from_bytes(f.read(4), "little")
            if vtype != 8:     # 8 = 字符串，architecture 一定是字符串
                return None
            vlen = int.from_bytes(f.read(8), "little")
            if "architecture" not in key:
                return None
            return f.read(vlen).decode("utf-8", "ignore")
    except Exception:
        return None


def run_check(model, quiet=False):
    print("\n===== 环境体检 =====")
    results = [check_python(), check_gpu(), check_llama_cpp()]
    results.append(check_model(model))
    print("====================\n")
    return all(results)


# ---------------- 2. 配置 ----------------
def load_config():
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception:
            warn("config.json 读不出来，用默认配置")
    return cfg


def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ---------------- 3. 运行 ----------------
def chat(llm):
    history = []
    print("输入问题开始对话，输入 q 退出，输入 /clear 清空记忆\n")
    while True:
        try:
            q = input("你: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            return
        if not q:
            continue
        if q.lower() == "q":
            print("再见！")
            return
        if q == "/clear":
            history.clear()
            print("[已清空对话记忆]\n")
            continue

        history.append({"role": "user", "content": q})
        print("AI: ", end="", flush=True)
        reply = ""
        try:
            for chunk in llm.create_chat_completion(messages=history, stream=True):
                t = chunk["choices"][0]["delta"].get("content", "")
                reply += t
                print(t, end="", flush=True)
        except ValueError as e:
            # 上下文超了：这是新手 100% 会遇到的报错，给他一句人话
            print(f"\n[上下文不够了] {e}")
            print("→ 输入 /clear 清空记忆，或用 --ctx 8192 重新启动\n")
            history.pop()
            continue
        print("\n")
        history.append({"role": "assistant", "content": reply})


def main():
    parser = argparse.ArgumentParser(description="一键部署本地大模型 v1.0（多模型版）")
    parser.add_argument("--model", help="模型别名（qwen/llama/glm）或 .gguf 路径")
    parser.add_argument("--ctx", type=int, help="上下文长度，如 4096/8192")
    parser.add_argument("--gpu-layers", type=int, help="放进 GPU 的层数，-1 为全部")
    parser.add_argument("--cpu", action="store_true", help="强制只用 CPU")
    parser.add_argument("--check", action="store_true", help="只做环境体检")
    parser.add_argument("--list", action="store_true", help="列出支持的模型")
    parser.add_argument("--download", metavar="别名", help="下载模型，如 --download glm")
    args = parser.parse_args()

    cfg = load_config()
    model = resolve_model(args.model or cfg.get("model"))

    if args.list:
        list_models(model["alias"])
        return

    if args.download:
        download_model(args.download)
        return

    # 已知跑不了的模型，别让用户等下载完、加载时才发现
    if model.get("unsupported"):
        fail(f"{model['name']} 用不了：{model['unsupported']}",
             "换个能跑的：python deploy.py --model qwen | llama | internlm")
        print()
        list_models()
        sys.exit(1)

    n_ctx = args.ctx or cfg["n_ctx"]
    n_gpu_layers = 0 if args.cpu else (
        args.gpu_layers if args.gpu_layers is not None else cfg["n_gpu_layers"]
    )

    if not run_check(model):
        print("体检没通过，按上面的提示解决后再运行。\n")
        sys.exit(1)
    if args.check:
        print("环境一切正常，可以直接运行：python deploy.py\n")
        return

    # 说明当前模式是怎么来的，避免"我明明没改，怎么变慢了"
    if args.cpu:
        print("本次强制 CPU 模式（临时生效，不写入配置）\n")
    elif cfg.get("n_gpu_layers") == 0:
        warn("配置文件里记的是纯 CPU 模式（n_gpu_layers=0）→ 想要显卡加速：--gpu-layers -1")
        print()

    # 显存够不够的提醒
    vram = detect_vram()
    if vram and n_gpu_layers != 0:
        est = 1000 + n_ctx * 0.046  # 粗估：权重 1GB + KV cache
        if est > vram * 0.92:
            warn(f"当前配置预计占用 {est:.0f} MiB，显存只有 {vram} MiB，可能爆显存")
            print("         → 降低 --ctx，或用 --gpu-layers 20 只放一部分层\n")

    print(f"模型：{model['name']}（{model['family']}）")
    print(f"加载中（ctx={n_ctx}, gpu_layers={n_gpu_layers}）...")
    try:
        prepare_cuda_dlls()
        from llama_cpp import Llama
        # 关键：chat_format 不同模型不一样。
        # None = 用 GGUF 里自带的对话模板；指定字符串 = 手动覆盖。
        # 模板错了不会报错，只会"答得很奇怪"——这是多模型适配最容易踩的坑。
        llm = Llama(model_path=model["path"], n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    chat_format=model.get("chat_format"), verbose=False)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        detail = str(e).lower()
        if "check_tensor_dims" in detail or "architecture" in detail:
            # 文件是好的，是引擎版本旧 → 说清楚，别让人以为是下载坏了
            arch = gguf_arch(model["path"]) or "未知"
            fail(f"引擎不认识这个模型的结构（{arch} 架构，文件本身没坏）",
                 "本机 llama-cpp-python 版本比模型旧。二选一：\n"
                 "         ① 升级引擎：pip install -U llama-cpp-python "
                 "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121\n"
                 "         ② 先跑能用的：python deploy.py --model qwen")
        else:
            fail(f"模型加载失败：{msg}",
                 "显存不够就降低 --ctx 或改用 --cpu；模板不对可换 --model 其它模型试试")
        sys.exit(1)

    # 只在命令行显式指定时才记住，避免临时参数被悄悄写死
    changed = {}
    if args.ctx is not None:
        changed["n_ctx"] = n_ctx
    if args.gpu_layers is not None:
        changed["n_gpu_layers"] = n_gpu_layers
    if args.model:
        changed["model"] = model["alias"] if model["alias"] != "custom" else model["file"]
    if changed:
        cfg.update(changed)
        save_config(cfg)
        print(f"已记住设置：{changed}（下次启动自动生效）")
    print("加载完成！\n")
    chat(llm)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中断")
