# -*- coding: utf-8 -*-
"""
compare.py —— 多模型横向对比（第 2 周产出）

用同一套代码、同一批问题，跑不同家族的模型，记录：
  - 加载时间
  - 生成速度（tok/s）
  - 回答内容（用来判断对话模板对不对）

用法：
  python compare.py                 # 跑所有已下载的模型
  python compare.py qwen llama      # 只跑指定的

为什么需要这个脚本：
  多模型适配最容易出的问题不是"报错"，而是"不报错但答得奇怪"——
  对话模板（chat template）用错了，模型会把你的问题当成普通文本续写，
  于是回一堆重复的废话。肉眼看速度看不出问题，必须对比输出内容。
"""
import json
import os
import sys
import time

import deploy  # 复用注册表：同一份配置，两个脚本共用

PROMPTS = [
    ("自我介绍", "用一句话介绍你自己"),
    ("写代码", "用 Python 写一个冒泡排序，只给代码，不要解释"),
    ("推理", "9.11 和 9.9 哪个大？一句话回答"),
]


def bench(model, prompt, n_ctx=2048):
    from llama_cpp import Llama

    t0 = time.time()
    llm = Llama(model_path=model["path"], n_ctx=n_ctx, n_gpu_layers=-1,
                chat_format=model.get("chat_format"), verbose=False)
    load_s = time.time() - t0

    t1 = time.time()
    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}], max_tokens=150
    )
    gen_s = time.time() - t1
    n_tok = out["usage"]["completion_tokens"]
    text = out["choices"][0]["message"]["content"].strip()

    del llm
    return {
        "load_s": round(load_s, 2),
        "tokens": n_tok,
        "tok_per_s": round(n_tok / gen_s, 1) if gen_s > 0 else 0,
        "text": text,
    }


def main():
    deploy.prepare_cuda_dlls()

    wanted = sys.argv[1:]
    models = []
    for m in deploy.MODEL_REGISTRY:
        e = deploy._entry(m)
        if wanted and e["alias"] not in wanted:
            continue
        if e.get("unsupported"):
            print(f"跳过 {e['alias']}：{e['unsupported']}")
            continue
        if not e["downloaded"]:
            print(f"跳过 {e['alias']}：文件未下载或不完整（{e['actual_mb']:.0f} MB）")
            continue
        models.append(e)

    if not models:
        print("没有可用的模型。先跑 python deploy.py --list 看缺哪个。")
        return

    # 读已有结果再合并：只跑一个模型时不会把其它模型的历史数据冲掉
    out_path = os.path.join(deploy.ROOT, "compare_results.json")
    results = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, encoding="utf-8") as f:
                results = json.load(f)
            print(f"已载入历史结果：{', '.join(results)}")
        except Exception:
            results = {}

    for m in models:
        print(f"\n{'=' * 52}\n{m['name']}（{m['family']}）\n{'=' * 52}")
        hits = {}
        for tag, prompt in PROMPTS:
            try:
                r = bench(m, prompt)
            except Exception as e:
                print(f"  [{tag}] 失败：{type(e).__name__}: {e}")
                continue
            hits[tag] = r
            print(f"  [{tag}] 加载 {r['load_s']}s | {r['tok_per_s']} tok/s | {r['tokens']} tok")
            preview = r["text"].replace("\n", " ")[:70]
            print(f"          → {preview}")
        results[m["alias"]] = {"name": m["name"], "family": m["family"], "runs": hits}

    # 汇总表
    print(f"\n{'=' * 52}\n汇总（单位 tok/s）\n{'=' * 52}")
    header = f"{'模型':<24}" + "".join(f"{t:<12}" for t, _ in PROMPTS)
    print(header)
    for alias, r in results.items():
        row = f"{r['name']:<24}"
        for t, _ in PROMPTS:
            v = r["runs"].get(t, {}).get("tok_per_s", "-")
            row += f"{str(v):<12}"
        print(row)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n明细已保存：{out_path}")


if __name__ == "__main__":
    main()
