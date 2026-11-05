# -*- coding: utf-8 -*-
"""
deploy.py v0.9 —— 一键部署脚本（新手可用版）

用法：
  python deploy.py                  # 自动检测显卡，推荐配置，直接开始对话
  python deploy.py --check          # 只做环境体检，不启动
  python deploy.py --ctx 8192       # 手动指定上下文长度
  python deploy.py --model 路径.gguf # 指定模型文件
  python deploy.py --cpu            # 强制用 CPU（没显卡或想对比速度时用）

设计原则（写给别人用的脚本，和写给自己用的脚本，区别就在这）：
  1. 路径以脚本自身为锚点，在哪个目录运行都能用
  2. 参数可配置，不写死在代码里
  3. 出错给人话，不甩一堆 traceback
  4. 先体检再运行，别等跑到一半才报错
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
CONFIG_PATH = os.path.join(ROOT, "config.json")

# 显存 -> (推荐模型, 推荐上下文) 的对照表
VRAM_TABLE = [
    (24000, "Qwen2.5-32B / 14B Q4", 8192),
    (12000, "Qwen2.5-14B / 7B Q4", 8192),
    (7000, "Qwen2.5-7B Q4 / 1.5B Q4", 8192),
    (4000, "Qwen2.5-1.5B Q4", 4096),
    (0, "Qwen2.5-0.5B Q4（很吃力）", 2048),
]

DEFAULT_CONFIG = {
    "model": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "n_ctx": 2048,
    "n_gpu_layers": -1,
    "n_threads": 0,
}


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
    try:
        import llama_cpp
        ok(f"llama-cpp-python {llama_cpp.__version__}")
        return True
    except ModuleNotFoundError:
        return fail(
            "没装 llama-cpp-python",
            'pip install llama-cpp-python --prefer-binary '
            '--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121',
        )
    except Exception as e:
        # 最常见：llama.dll 缺依赖（缺 CUDA 运行库）
        return fail(
            f"库装上了但加载失败：{type(e).__name__}",
            "大概率缺 CUDA 运行库：pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12\n"
            "         再把 nvidia/*/bin 里的 cudart64_12.dll、cublas64_12.dll、cublasLt64_12.dll\n"
            "         复制到 site-packages/llama_cpp/lib/",
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


def check_model(path):
    if os.path.exists(path):
        size = os.path.getsize(path) / 1024 / 1024
        ok(f"模型文件 {os.path.basename(path)}（{size:.0f} MB）")
        return True
    name = os.path.basename(path)
    return fail(f"模型不存在：{path}", f"先下载：\n         curl -L -o models/{name} "
                                       f"https://hf-mirror.com/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/{name}")


def run_check(model_path, quiet=False):
    print("\n===== 环境体检 =====")
    results = [check_python(), check_gpu(), check_llama_cpp()]
    results.append(check_model(model_path))
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
    parser = argparse.ArgumentParser(description="一键部署本地大模型 v0.9")
    parser.add_argument("--model", help="模型文件路径")
    parser.add_argument("--ctx", type=int, help="上下文长度，如 4096/8192")
    parser.add_argument("--gpu-layers", type=int, help="放进 GPU 的层数，-1 为全部")
    parser.add_argument("--cpu", action="store_true", help="强制只用 CPU")
    parser.add_argument("--check", action="store_true", help="只做环境体检")
    args = parser.parse_args()

    cfg = load_config()
    model_path = args.model or os.path.join(MODELS_DIR, cfg["model"])
    if args.model and not os.path.isabs(args.model):
        model_path = os.path.abspath(args.model)
    n_ctx = args.ctx or cfg["n_ctx"]
    n_gpu_layers = 0 if args.cpu else (
        args.gpu_layers if args.gpu_layers is not None else cfg["n_gpu_layers"]
    )

    if not run_check(model_path):
        print("体检没通过，按上面的提示解决后再运行。\n")
        sys.exit(1)
    if args.check:
        print("环境一切正常，可以直接运行：python deploy.py\n")
        return

    # 显存够不够的提醒
    vram = detect_vram()
    if vram and n_gpu_layers != 0:
        est = 1000 + n_ctx * 0.046  # 粗估：权重 1GB + KV cache
        if est > vram * 0.92:
            warn(f"当前配置预计占用 {est:.0f} MiB，显存只有 {vram} MiB，可能爆显存")
            print("         → 降低 --ctx，或用 --gpu-layers 20 只放一部分层\n")

    print(f"加载模型（ctx={n_ctx}, gpu_layers={n_gpu_layers}）...")
    try:
        from llama_cpp import Llama
        llm = Llama(model_path=model_path, n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers, verbose=False)
    except Exception as e:
        fail(f"模型加载失败：{type(e).__name__}: {e}",
             "显存不够就降低 --ctx 或改用 --cpu；其他错误看上面的体检结果")
        sys.exit(1)

    cfg.update({"model": os.path.basename(model_path), "n_ctx": n_ctx,
                "n_gpu_layers": n_gpu_layers})
    save_config(cfg)
    print("加载完成！\n")
    chat(llm)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中断")
