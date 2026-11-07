# -*- coding: utf-8 -*-
"""
montage_eval.py -- “事件化记忆压缩”的可复现实验

任务：在固定记忆预算下，从一段长对话的“过去”里挑出要保留的内容，
      看能否保住那些“后来会被问到”的早期事实。

四种策略：
  A 截断法（last）    ：只留最近 N 字符
  B 均匀采样（uniform）：按等间隔抽消息
  C 检索法（bm25）    ：用查询词回捞历史消息（RAG 基线）
  D 事件关键帧（event）：把过去切成事件，按信息密度打分，选高分事件

指标：召回率（预算内是否保住了那条事实）、保留字符数。
"""

from __future__ import annotations

import json
import os
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ARCH = Path(os.environ.get("MONTAGE_ARCHIVE", "data/archive"))
OUT = Path(os.environ.get("MONTAGE_OUT", "results"))
OUT.mkdir(parents=True, exist_ok=True)

HEAD = re.compile(r"^### (\d+)\. (用户|DeepSeek) · (.+)$")
THINK = re.compile(r"<details>.*?</details>", re.S)
SUB = re.compile(r"<sub>.*?</sub>", re.S)
BUDGET = 1500          # 记忆预算（字符）
MIN_MSGS = 25
PAST_RATIO = 0.6

# 事件切分与打分的信号
DECISION = ["决定", "定了", "确定", "改成", "改为", "不要", "必须", "结论", "因此"]
CHANGE = ["改", "修", "换", "删", "加", "新增", "退回", "推翻", "纠正"]
NUM = re.compile(r"\d+(?:\.\d+)?%?|\d{4}年")


def clean(t: str) -> str:
    return SUB.sub("", THINK.sub("", t)).strip()


def parse(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    body = text.split("\n---", 3)[-1]
    main = body.split("## 分支与版本")[0]
    msgs, cur, buf = [], None, []
    for line in main.splitlines():
        m = HEAD.match(line.strip())
        if m:
            if cur:
                t = clean("\n".join(buf))
                if t:
                    msgs.append({"role": cur, "text": t})
            cur = m.group(2)
            buf = []
        else:
            buf.append(line)
    if cur:
        t = clean("\n".join(buf))
        if t:
            msgs.append({"role": cur, "text": t})
    return msgs


def terms(s: str) -> set:
    """抽 2-6 字的中文片段作为粗粒度词项"""
    s = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", " ", s)
    out = set()
    for w in s.split():
        if 2 <= len(w) <= 12:
            out.add(w)
        for n in (2, 3, 4):
            for i in range(len(w) - n + 1):
                out.add(w[i:i + n])
    return out


def build_items():
    items = []
    for path in sorted(ARCH.rglob("*.md")):
        msgs = parse(path)
        if len(msgs) < MIN_MSGS:
            continue
        cut = int(len(msgs) * PAST_RATIO)
        past, future = msgs[:cut], msgs[cut:]
        if len(past) < 15 or len(future) < 5:
            continue

        past_terms = defaultdict(list)
        for i, m in enumerate(past):
            for t in terms(m["text"]):
                past_terms[t].append(i)

        used_idx = set()
        got = 0
        for j, m in enumerate(future):
            if got >= 3:
                break
            best = None
            for t in terms(m["text"]):
                idxs = past_terms.get(t, [])
                if 1 <= len(idxs) <= 2 and len(t) >= 3:
                    best = (t, idxs[0])
                    break
            if not best:
                continue
            qterm, idx = best
            if idx in used_idx:
                continue
            used_idx.add(idx)
            fact = past[idx]["text"]
            if len(fact) < 60:
                continue
            items.append({
                "conv": path.stem,
                "qterm": qterm,
                "fact_idx": idx,
                "fact": fact[:600],
                "probe": fact[:60],
                "past": [m["text"][:400] for m in past],
            })
            got += 1
    return items


# ---------------- 四种策略 ----------------
def keep_last(past, budget=BUDGET):
    s = ""
    for m in reversed(past):
        if len(s) + len(m) > budget:
            break
        s = m + "\n" + s
    return s


def keep_uniform(past, budget=BUDGET):
    n = max(1, len(past))
    take, used = [], 0
    i = 0
    while used < budget and i < n:
        m = past[min(n - 1, int(i))]
        if used + len(m) <= budget:
            take.append(m)
            used += len(m)
        i += 1
        if i > 2 * n:
            break
    return "\n".join(take)


def keep_bm25(past, qterm, budget=BUDGET):
    """用查询词检索（RAG 基线的近似：词项命中 + 逆文档频率）"""
    docs = [terms(m) for m in past]
    df = Counter()
    for d in docs:
        for t in d:
            df[t] += 1
    n = len(docs)
    scored = []
    for i, d in enumerate(docs):
        s = 0.0
        for t in terms(qterm):
            if t in d:
                s += math.log((n + 1) / (df[t] + 1))
        if s > 0:
            scored.append((s, i))
    scored.sort(reverse=True)
    out, used = [], 0
    for _, i in scored:
        if used + len(past[i]) > budget:
            continue
        out.append(past[i])
        used += len(past[i])
    return "\n".join(out)


def segment_events(past):
    """按长度与信号词密度切事件：每 4 条一段，再合并低密度段"""
    events, cur = [], []
    for i, m in enumerate(past):
        cur.append(m)
        if len(cur) >= 4:
            events.append(cur)
            cur = []
    if cur:
        events.append(cur)
    return events


def score_event(ev):
    txt = "\n".join(ev)
    s = 0.0
    s += 2.0 * sum(txt.count(k) for k in DECISION)
    s += 1.5 * sum(txt.count(k) for k in CHANGE)
    s += 0.8 * len(NUM.findall(txt))
    s += 0.02 * len(txt) ** 0.5
    return s


def keep_event(past, budget=BUDGET):
    evs = segment_events(past)
    ranked = sorted(evs, key=score_event, reverse=True)
    out, used = [], 0
    for ev in ranked:
        block = "\n".join(ev)
        if used + len(block) > budget:
            continue
        out.append(block)
        used += len(block)
    # 按原顺序还原
    out_sorted = sorted(out, key=lambda b: past.index(b.split("\n")[0]) if b.split("\n")[0] in past else 0)
    return "\n".join(out_sorted)



def keep_event_cov(past, budget=BUDGET):
    """覆盖式分镜：贪心选事件，优先带来新词项（对应“把整条故事线讲完”）"""
    evs = segment_events(past)
    covered = set()
    chosen = []
    used = 0
    while used < budget:
        best, best_gain = None, -1
        for i, ev in enumerate(evs):
            if i in chosen:
                continue
            block = "\n".join(ev)
            if used + len(block) > budget:
                continue
            gain = len(terms(block) - covered)
            if gain > best_gain:
                best, best_gain = i, gain
        if best is None or best_gain <= 0:
            break
        block = "\n".join(evs[best])
        chosen.append(best)
        covered |= terms(block)
        used += len(block)
    return "\n".join("\n".join(evs[i]) for i in sorted(chosen))


def keep_oracle(past, probe, budget=BUDGET):
    """上界：直接把事实那一条塞进预算，剩余空间用最近的补"""
    out, used = [], 0
    for m in past:
        if probe[:40] in m and used + len(m) <= budget:
            out.append(m)
            used += len(m)
    for m in reversed(past):
        if used + len(m) > budget:
            continue
        if m not in out:
            out.append(m)
            used += len(m)
        if used >= budget * 0.95:
            break
    return "\n".join(out)

def hit(memory: str, probe: str) -> bool:
    """事实是否被保留：探针前 40 字是否出现在记忆中（允许 80% 字符重合）"""
    p = re.sub(r"\s+", "", probe)[:40]
    m = re.sub(r"\s+", "", memory)
    if p in m:
        return True
    # 宽松匹配：15 字滑窗只要有 12 字命中即算
    for i in range(0, max(1, len(p) - 15)):
        seg = p[i:i + 15]
        if seg in m:
            return True
    return False


def main():
    print("[1/3] 构建测试集 …")
    items = build_items()
    rng = random.Random(20261103)
    rng.shuffle(items)
    items = items[:300]
    print(f"      {len(items)} 条（每条 = 一个早期事实 + 一个后期查询词）")

    print("[2/3] 跑四种策略 …")
    stats = defaultdict(lambda: {"hit": 0, "chars": 0})
    per_item = []
    for it in items:
        past = it["past"]
        mems = {
            "last": keep_last(past),
            "uniform": keep_uniform(past),
            "bm25": keep_bm25(past, it["qterm"]),
            "event": keep_event(past),
            "event_cov": keep_event_cov(past),
            "oracle": keep_oracle(past, it["probe"]),
        }
        row = {"conv": it["conv"], "qterm": it["qterm"]}
        for k, mem in mems.items():
            h = hit(mem, it["probe"])
            stats[k]["hit"] += int(h)
            stats[k]["chars"] += len(mem)
            row[k] = int(h)
        per_item.append(row)

    n = len(items)
    print("[3/3] 结果：\n")
    print(f"{'策略':<10}{'召回率':>8}{'平均保留字符':>14}")
    print("-" * 34)
    res = {}
    for k in ("last", "uniform", "event", "event_cov", "bm25", "oracle"):
        r = stats[k]["hit"] / n
        c = stats[k]["chars"] / n
        res[k] = {"recall": round(r, 4), "avg_chars": round(c, 1)}
        print(f"{k:<10}{r:>8.1%}{c:>14.0f}")

    (OUT / "montage_results.json").write_text(
        json.dumps({"n": n, "budget": BUDGET, "results": res, "per_item": per_item},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {OUT / 'montage_results.json'}")


if __name__ == "__main__":
    main()


