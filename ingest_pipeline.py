"""
电商数据集摄入 Pipeline

把电商数据集（商品、评论、客服记录）处理成适合摄入的格式，
批量写入 wiki_knowledge/raw/，再调用 WikiKnowledgeBase.ingest()。

使用方式：
  python ingest_pipeline.py --source ./data/ecommerce_dataset.csv --type review
  python ingest_pipeline.py --source ./data/products.json --type product
  python ingest_pipeline.py --demo   # 用内置示例数据快速演示
  python ingest_pipeline.py --source ./JDDC/chat.txt

数据集推荐（CSDN 博客列出的电商数据集）：
  - 亚马逊商品评论数据集（CSV格式，含商品名/评分/评论文本）
  - 电商客服对话数据集（含用户问题/客服回答）
  - 商品信息数据集（含品类/价格/描述）
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# 把 python-impl 加入路径
import sys
sys.path.insert(0, str(Path(__file__).parent))

from memory.wiki_knowledge_base import WikiknowledgeBase


# ── 数据适配器：把各种格式转为统一文本 ───────────────────────────────────────

def load_csv_reviews(file_path: str, limit: int = 50) -> list[str]:
    """
    加载 CSV 格式的用户评论数据。
    兼容常见列名：review_body / reviewText / comment / text
    """
    rows = []
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            # 兼容多种列名
            text = (
                row.get("review_body")
                or row.get("reviewText")
                or row.get("comment")
                or row.get("text")
                or row.get("review")        # online_shopping_10_cats 数据集
                or ""
            )
            product = (
                row.get("product_title")
                or row.get("product_name")
                or row.get("asin")
                or row.get("cat")           # online_shopping_10_cats 用品类作为"商品"
                or "未知商品"
            )
            rating = row.get("star_rating") or row.get("overall") or row.get("rating") or ""
            # label: 1=好评 0=差评（该数据集没有星级，用情感标签代替评分）
            if not rating and row.get("label") in ("0", "1"):
                rating = "好评" if row["label"] == "1" else "差评"
            if text:
                rows.append(
                    f"商品：{product}\n评分：{rating}\n评论：{text}"
                )
    return rows


def load_csv_products(file_path: str, limit: int = 30) -> list[str]:
    """加载商品信息 CSV"""
    rows = []
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            rows.append(json.dumps(dict(row), ensure_ascii=False))
    return rows


def load_json_records(file_path: str, limit: int = 50) -> list[str]:
    """加载 JSON 格式的数据（支持 JSON Lines 和普通 JSON 数组）"""
    records = []
    with open(file_path, encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                records = data[:limit]
            elif isinstance(data, dict):
                records = [data]
        except json.JSONDecodeError:
            # 尝试 JSON Lines 格式
            f.seek(0)
            for i, line in enumerate(f):
                if i >= limit:
                    break
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return [json.dumps(r, ensure_ascii=False) for r in records]


def load_dialogue_corpus(file_path: str, limit: int = 50) -> list[str]:
    """
    加载 E-commerce Dialogue Corpus 格式的多轮客服对话。

    格式：label \\t 对话轮次1 \\t 轮次2 \\t ... \\t 候选回复
      label=1 表示候选回复是正确回复，0 为负样本。

    处理：只取 label=1 的正样本，去除分词空格还原自然句子，
    把每段会话整理成"用户/客服"交替的对话文本。

    用流式逐行读取，避免大文件（train.txt 257MB）一次性载入内存。
    """
    rows = []
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if len(rows) >= limit:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or parts[0] != "1":
                continue
            utterances = parts[1:]          # 对话轮次 + 最后一句候选回复
            # 去除分词空格："要 买 一把" -> "要买一把"
            cleaned = ["".join(u.split()) for u in utterances if u.strip()]
            if len(cleaned) < 2:
                continue
            # 交替标注角色：奇数位为用户，偶数位为客服（最后一句是正确客服回复）
            lines = []
            for idx, utt in enumerate(cleaned):
                role = "用户" if idx % 2 == 0 else "客服"
                lines.append(f"{role}：{utt}")
            rows.append("【客服对话】\n" + "\n".join(lines))
    return rows


def load_jddc_chat(file_path: str, limit: int = 50) -> list[str]:
    """
    加载 JDDC（京东客服对话语料）chat.txt 格式。

    制表符分隔，7 列：
      session_id, user_id, waiter_send, is_transfer, is_repeat, content, (空列)
      waiter_send: 0=用户 1=客服

    按 session 聚合成完整多轮对话，用 waiter_send 精确标注角色，
    合并同一说话人的连续发言。流式逐行读，避免大文件一次性载入。
    """
    rows = []
    cur_sid = None
    turns: list[tuple[str, str]] = []   # [(role, text), ...]

    def flush(turns):
        # 合并同一角色连续发言，整理成对话文本
        merged: list[tuple[str, str]] = []
        for role, text in turns:
            if merged and merged[-1][0] == role:
                merged[-1] = (role, merged[-1][1] + " " + text)
            else:
                merged.append((role, text))
        if len(merged) < 2:
            return None
        lines = [f"{r}：{t}" for r, t in merged]
        return "【客服对话】\n" + "\n".join(lines)

    with open(file_path, encoding="utf-8", errors="ignore") as f:
        f.readline()    # 跳过表头
        for line in f:
            if len(rows) >= limit:
                break
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 7:
                continue
            # 注意：数据行的 content 实际落在第 6 列（列5为空，表头与数据错位一列）
            sid, waiter, content = cols[0], cols[2], cols[6].strip()
            if not content:
                continue
            if sid != cur_sid:
                if cur_sid is not None:
                    dialog = flush(turns)
                    if dialog:
                        rows.append(dialog)
                cur_sid = sid
                turns = []
            role = "客服" if waiter == "1" else "用户"
            turns.append((role, content))
        # 收尾最后一个 session
        if len(rows) < limit and turns:
            dialog = flush(turns)
            if dialog:
                rows.append(dialog)
    return rows


# ── 示例数据（--demo 模式用） ────────────────────────────────────────────────

DEMO_PRODUCTS = """
商品名称：无线蓝牙耳机 ProX
品类：数码配件
价格：299元
描述：主动降噪，续航30小时，IPX4防水，支持多设备连接
保修：1年
退换货：7天无理由退换，需保持包装完整

商品名称：智能手环 V3
品类：智能穿戴
价格：199元
描述：心率监测、睡眠分析、50米防水、14天续航
保修：1年
退换货：7天无理由退换

商品名称：便携充电宝 20000mAh
品类：数码配件
价格：159元
描述：20000mAh大容量，双向快充，支持3设备同时充电
保修：6个月
退换货：7天无理由退换，不支持已激活产品退换
"""

DEMO_REVIEWS = """
用户评论数据集（电商平台真实用户反馈）：

[评论1]
商品：无线蓝牙耳机 ProX
评分：5/5
内容：降噪效果很好，通勤必备！连接稳定，没有延迟。就是充电盒有点大。

[评论2]
商品：无线蓝牙耳机 ProX
评分：2/5
内容：买了两周左右右耳声音变小，联系客服说要寄回去检测，等了10天才修好寄回来。
客服处理速度太慢了，希望改进。

[评论3]
商品：智能手环 V3
评分：4/5
内容：心率监测挺准的，睡眠数据也有参考价值。但APP有时候同步失败，需要重新配对。

[评论4]
商品：便携充电宝 20000mAh
评分：1/5
内容：收到后发现是翻新品，容量严重虚标，实测只有8000mAh左右。申请退款，
客服先说7天退换政策不支持已激活产品，后来升级投诉才给退了。

[评论5]
商品：智能手环 V3
评分：3/5
内容：续航确实能到14天，但防水只有50米，我游泳用了几次就出问题了。建议标清楚
适用场景。
"""

DEMO_FAQ = """
客服常见问题数据集：

Q: 我的耳机右耳没声音了怎么办？
A: 您好，请先尝试以下步骤：1) 将耳机放回充电盒充电10分钟后重试；2) 在蓝牙设置里
删除设备重新配对；3) 如问题持续，可申请售后检测，保修期内免费维修，超出保修期
检测费50元。

Q: 申请退款多久到账？
A: 退款审核通过后，原路返回：支付宝/微信1-3个工作日，银行卡3-7个工作日。
如超过7个工作日未到账，请联系客服提供订单号核查。

Q: 商品保修怎么办理？
A: 保修期内（购买之日起1年）出现非人为损坏，免费维修或换新。办理方式：
1) 在APP订单页点"申请售后"；2) 选择"维修"；3) 按提示寄回商品（我们承担来回运费）。

Q: 充电宝容量和宣传不符怎么投诉？
A: 非常抱歉给您带来不好的体验。请提供：1) 订单号；2) 实测视频或截图。
我们将在24小时内核查，确认属实将按虚假宣传处理：退款+补偿购物券。

Q: 手环防水是游泳级别吗？
A: 智能手环 V3 防水等级为 IPX7（50米静态防水），适合洗手、淋浴等日常防水场景。
不建议用于游泳、潜水等持续浸水场景。如在游泳场景损坏，属于非保修范围。
"""


# ── 主 Pipeline ──────────────────────────────────────────────────────────────

async def run_pipeline(
    wiki:        WikiknowledgeBase,
    data:        str,
    source_type: str,
    source_name: str,
) -> dict:
    """把一批数据存到 raw/ 再摄入 Wiki"""
    # 保存到 raw/
    raw_subdir = {
        "product": wiki.raw_dir / "products",
        "review":  wiki.raw_dir / "reviews",
        "order":   wiki.raw_dir / "orders",
        "faq":     wiki.raw_dir / "orders",
    }.get(source_type, wiki.raw_dir)

    raw_file = raw_subdir / f"{source_name}.md"
    raw_file.write_text(data, encoding="utf-8")
    print(f"✓ 原始数据保存至 {raw_file}")

    # 摄入 Wiki
    print(f"  正在摄入（LLM 分析中）...")
    result = await wiki.ingest(str(raw_file), source_type)
    print(f"  ✓ 创建 {result.get('pages_created', 0)} 页，更新 {result.get('pages_updated', 0)} 页")
    return result


async def main():
    parser = argparse.ArgumentParser(description="电商数据集摄入 Pipeline")
    parser.add_argument("--source", help="数据文件路径（CSV/JSON）")
    parser.add_argument("--type",   default="auto",
                        choices=["product", "review", "order", "faq", "auto"],
                        help="数据类型")
    parser.add_argument("--limit",  type=int, default=50,
                        help="最多处理条数（避免摄入过多）")
    parser.add_argument("--demo",   action="store_true",
                        help="使用内置示例数据快速演示")
    parser.add_argument("--lint",   action="store_true",
                        help="摄入后执行健康检查")
    args = parser.parse_args()

    wiki = WikiknowledgeBase(
        llm=ChatOpenAI(
            model=os.getenv("MODEL_NAME", "gpt-4o"),
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            streaming=True,
        )
    )

    print("=" * 50)
    print("  LLM Wiki 知识库摄入 Pipeline")
    print("=" * 50)

    if args.demo:
        # 演示模式：摄入三类示例数据
        print("\n[演示模式] 摄入内置示例数据...\n")

        for name, data, stype in [
            ("demo_products", DEMO_PRODUCTS, "product"),
            ("demo_reviews",  DEMO_REVIEWS,  "review"),
            ("demo_faq",      DEMO_FAQ,       "faq"),
        ]:
            print(f"▶ 摄入 {name} ({stype})")
            await run_pipeline(wiki, data, stype, name)
            print()

    elif args.source:
        source_path = Path(args.source)
        if not source_path.exists():
            print(f"错误：文件不存在 {args.source}")
            return

        print(f"\n▶ 加载数据：{args.source}")

        # 根据文件格式加载
        suffix = source_path.suffix.lower()
        if suffix == ".csv":
            if args.type in ("review", "auto"):
                rows = load_csv_reviews(args.source, args.limit)
            else:
                rows = load_csv_products(args.source, args.limit)
            data = "\n\n".join(rows)
        elif suffix in (".json", ".jsonl"):
            rows = load_json_records(args.source, args.limit)
            data = "\n\n".join(rows)
        elif suffix == ".txt":
            # 嗅探格式：JDDC chat.txt 带制表符表头，否则按 E-commerce 对话语料处理
            with open(source_path, encoding="utf-8", errors="ignore") as _f:
                first_line = _f.readline()
            if first_line.startswith("session_id\t"):
                rows = load_jddc_chat(args.source, args.limit)
            else:
                rows = load_dialogue_corpus(args.source, args.limit)
            data = "\n\n".join(rows)
        else:
            data = source_path.read_text(encoding="utf-8")

        print(f"  加载了 {len(data)} 字符的数据")
        await run_pipeline(wiki, data, args.type, source_path.stem)

    else:
        print("请指定 --source 或 --demo")
        parser.print_help()
        return

    # 健康检查
    if args.lint:
        print("\n▶ 执行健康检查...")
        lint_result = await wiki.lint()
        print(f"  检查了 {lint_result['pages_checked']} 个页面")
        print(f"  发现 {lint_result['issues_found']} 个问题")
        for issue in lint_result["issues"]:
            print(f"  [{issue['severity'].upper()}] {issue['page']}: {issue['issues']}")

    # 打印统计
    stats = wiki.get_stats()
    print("\n📊 知识库统计：")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n✅ 完成！Wiki 文件位于：")
    print(f"  {Path(__file__).parent / 'wiki_knowledge' / 'wiki'}")


if __name__ == "__main__":
    asyncio.run(main())