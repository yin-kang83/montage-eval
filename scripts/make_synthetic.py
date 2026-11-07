# -*- coding: utf-8 -*-
"""Generate a small synthetic dialogue archive for montage-eval smoke tests."""
from __future__ import annotations

import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "archive"
OUT.mkdir(parents=True, exist_ok=True)
rng = random.Random(20261107)

TOPICS = ["预算", "折旧", "租赁", "税务", "审计", "现金流", "存货", "收入"]


def msg(i, role, text):
    return f"### {i}. {role} · turn {i}\n{text}"


def make_conv(idx):
    token = f"ZQ{idx:03d}"
    n = 30
    lines = [f"# Synthetic conversation {idx}"]
    for i in range(1, n + 1):
        role = "用户" if i % 2 == 1 else "DeepSeek"
        if i == 5:
            text = (f"我们决定把 {token} 项目的预算定为 1500 万元，"
                    f"分三期支付，第一期为 500 万元，第二期为 600 万元，第三期为 400 万元。")
        elif i == 25:
            text = f"请回顾 {token} 项目的预算总额和分期安排。"
        else:
            filler = ("这一段是背景说明，包含假设、口径、待办和无关细节，用来把对话拉长。"
                      "我们保留上下文连贯性，但不引入新的关键结论。" * 3)
            text = (f"这是第 {i} 轮讨论，主题是{TOPICS[i % len(TOPICS)]}。"
                    f"我们记录一些背景、假设、口径和待办事项，用于后续核对。" + filler)
        lines.append(msg(i, role, text))
    (OUT / f"synthetic_{idx:02d}.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    for i in range(1, 13):
        make_conv(i)
    print("wrote", len(list(OUT.glob("*.md"))), "conversations to", OUT)
