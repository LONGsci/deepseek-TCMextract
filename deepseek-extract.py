import asyncio
import pandas as pd
import backoff
from openai import AsyncOpenAI
import logging
import re
import os
import random
from typing import List, Dict, Any
import time
from datetime import timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Timer:
    """运行时长统计器"""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()
        logger.info("▶ 开始处理，计时启动")

    def stop(self):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        logger.info(f"■ 处理完成，总耗时: {str(timedelta(seconds=elapsed))}")

    def get_elapsed(self):
        if self.start_time is None:
            return "未启动"
        current = self.end_time if self.end_time else time.time()
        return str(timedelta(seconds=current - self.start_time))

# DeepSeek API配置
API_KEY = ""  # 替换为实际API密钥
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

# 系统提示词
SYSTEM_PROMPT = """
作为中医文献专家，请严格按以下要求处理文本，严格按照输出格式输出结果：

【处理步骤】
1. 判断是否为包含穴位名称和病症名称的针灸文献词条：
        - 满足以下任一即为有效：
            (1) 含穴位名称 + (引导词后病症 或 独立症状行)
            (2) 符合"症状行-穴位行"交替结构
        -对应输出格式中的“是否为有效词条”（是/否），若为无效词条则跳过下述步骤。
2. 若是则：
   a. 检查文本是否包含多个处方（如多个古籍引用、兼xx穴治xxx、"症状描述+穴位列表"行交替模式）。
        -如果是，拆分为多条文本；否则，将文本视为单条。
        -如果涉及到“兼xx穴治xxx”的拆分，那么拆分后应补充整段中第一个出现的穴位名称（A穴），如“A穴兼xx穴治xxx（出处）”
        -如果存在连续行交替结构也需要拆分：
          - 奇数行为症状描述（无穴位名）
          - 偶数行为穴位列表（空格分隔的穴位名）
          - 出现分行符且包含多个独立病症-穴位组合
          - 每两行（症状行+穴位行）组成独立处方
        - * 拆分后每条需独立完成后续提取
   b. 对每条拆分后的文本（或单条文本）提取以下内容：
      - 病症名称：从"主治"或"治"等提示词后提取病症。若遇连续症状描述（如"身重善饥肉痿"），必须拆分为独立短词语（2-4字中医术语），用分号分隔。
        （拆分原则：按中医症状单元划分，如"身重"、"善饥"、"肉痿"、"足不收"、"行善瘛"、"脚下痛"）
      - 刺法：仅提取刺法类型术语，排除所有操作参数：正确："燔针刺"、"关刺"、“透天凉”等针刺操作。错误："针五分"、"刺入三分"（属针刺深度）。无明确刺法类型时输出NA
      - 灸法：仅提取灸法类型术语，排除所有操作参数：正确："豉饼灸"、"隔姜灸"、“隔饼灸”、“化脓灸”等灸法操作。错误："灸三壮"、"灸七炷"（属艾灸壮数）。单纯"灸"字无具体方法时输出NA
      - 舌象：必须含舌部描述词："舌干"/"舌绛"/"舌上苔"，排除非特异性描述如"口苦"
      - 穴位名称：所有出现的穴位名（非经络名称，如"足太阳"等），字段前后都有可能出现（注意“兼”之后可能存在穴位)
      - 治法："寒下""泻肝"等中医治疗方法
      - 艾灸壮数：格式"穴位 壮数"，多个用分号分隔，使用原文中的中文数字而不是阿拉伯数字
      - 针刺深度：格式"穴位 深度" （xx x寸x分），多个用分号分隔，使用原文中的中文数字而不是阿拉伯数字
      - 留针时长：格式"穴位 时间"（xx x呼），多个用分号分隔，使用原文中的中文数字而不是阿拉伯数字
   c. 输出时，如果有多条文本（如多个古籍引用），则生成多个独立记录（每条记录对应一个拆分后的文本）；如果只有单条文本，则生成一个记录。
   
【输出格式】
输出一个或多个记录（根据拆分情况，多个记录间以"---"分隔），每个记录格式如下：
是否为有效词条：是/否
是否拆分词条：是/否
病症名称：[值/NA]
刺法：[值/NA]   // 禁止含操作参数
灸法：[值/NA]   // 禁止含操作参数
舌象：[值/NA]
穴位名称：[值/NA]
治法：[值/NA]
艾灸壮数：[值/NA]
针刺深度：[值/NA]
留针时长：[值/NA]
处方拆分：[当前处理的文本片段/NA] // 对于拆分后的文本，存储该片段；未拆分时保留NA]
"""


# 重试装饰器（指数退避策略）
@backoff.on_exception(backoff.expo, Exception, max_tries=5, max_time=30)
async def get_deepseek_response(client: AsyncOpenAI, text: str) -> str:
    """调用DeepSeek API并处理重试逻辑"""
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"请处理以下中医文献文本：\n{text}"}
            ],
            temperature=0.1,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"API调用失败: {str(e)}")
        raise


def parse_api_response(response: str) -> List[Dict[str, str]]:
    """解析API返回的文本为结构化数据"""
    records = []
    # 分割多个记录
    record_blocks = re.split(r'-{3,}', response)

    for block in record_blocks:
        if not block.strip():
            continue

        record = {
            "是否为有效词条": "NA",
            "是否拆分词条": "NA",
            "病症名称": "NA",
            "刺法": "NA",
            "灸法": "NA",
            "舌象": "NA",
            "穴位名称": "NA",
            "治法": "NA",
            "艾灸壮数": "NA",
            "针刺深度": "NA",
            "留针时长": "NA",
            "处方拆分": "NA"
        }

        # 解析键值对
        for line in block.strip().split('\n'):
            # 处理中文/英文冒号
            if '：' in line:
                parts = line.split('：', 1)
            elif ':' in line:
                parts = line.split(':', 1)
            else:
                continue

            if len(parts) < 2:
                continue

            key = parts[0].strip()
            value = parts[1].strip()

            # 匹配已知字段
            if key in record:
                record[key] = value

        records.append(record)

    return records


async def process_text(client: AsyncOpenAI, row: Dict, idx: int) -> List[Dict[str, Any]]:
    """处理单个文本条目"""
    try:
        logger.info(f"正在处理条目 {idx}")
        text = row["关键词所在章节对应的正文"]
        response = await get_deepseek_response(client, text)

        # 添加随机延迟避免API限流 (0.1-0.5秒)
        delay = random.uniform(0.1, 0.5)
        await asyncio.sleep(delay)

        records = parse_api_response(response)

        # 添加原始数据信息
        for record in records:
            # 添加原始行数据
            record["文献索引号"] = idx
            record["关键词所在章节"] = row.get("关键词所在章节", "")
            record["关键词所在章节对应的正文"] = text

            # 添加处理状态
            record["处理状态"] = "成功"

        return records
    except Exception as e:
        logger.error(f"处理条目 {idx} 失败: {str(e)}")
        # 返回错误标记记录，保留原始数据
        return [{
            "文献索引号": idx,
            "关键词所在章节": row.get("关键词所在章节", ""),
            "关键词所在章节对应的正文": row.get("关键词所在章节对应的正文", ""),
            "处理状态": f"失败: {str(e)}"
        }]


def load_progress(progress_file: str) -> int:
    """加载处理进度"""
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                return int(f.read().strip())
        except:
            logger.warning("进度文件损坏，将从头开始处理")
    return 0


def save_progress(progress_file: str, index: int):
    """保存处理进度"""
    with open(progress_file, "w") as f:
        f.write(str(index))


def save_partial_results(results: List[Dict], output_file: str):
    """保存部分结果"""
    if not results:
        return

    # 定义列顺序
    column_order = [
        "文献索引号", "关键词所在章节", "关键词所在章节对应的正文",
        "是否为有效词条", "是否拆分词条", "病症名称",
        "刺法", "灸法", "舌象", "穴位名称", "治法",
        "艾灸壮数", "针刺深度", "留针时长", "处方拆分", "处理状态"
    ]

    result_df = pd.DataFrame(results)


    for col in column_order:
        if col not in result_df.columns:
            result_df[col] = ""

    # 按指定顺序排列列
    result_df = result_df[column_order]

    # 如果文件已存在，追加数据
    if os.path.exists(output_file):
        existing_df = pd.read_excel(output_file)
        # 确保列顺序一致
        existing_df = existing_df[column_order]
        result_df = pd.concat([existing_df, result_df], ignore_index=True)

    result_df.to_excel(output_file, index=False)
    logger.info(f"已保存部分结果至: {output_file}")


def create_empty_record(row: Dict, idx: int, reason: str) -> Dict:
    """创建空记录占位"""
    return {
        "文献索引号": idx,
        "关键词所在章节": row.get("关键词所在章节", ""),
        "关键词所在章节对应的正文": "",
        "处理状态": reason
    }

async def process_batch(tasks: List, output_file: str, progress_file: str, progress_idx: int):
    """处理批次任务并保存结果"""
    results = await asyncio.gather(*tasks)
    flat_results = [record for sublist in results for record in sublist]
    save_partial_results(flat_results, output_file)
    save_progress(progress_file, progress_idx)

def log_progress(processed: int, total: int, timer: Timer):
    """打印处理进度"""
    progress = processed / total * 100
    speed = processed / max(1, (time.time() - timer.start_time))
    logger.info(
        f"📌 进度: {processed}/{total} ({progress:.1f}%) | "
        f"速度: {speed:.2f}条/秒 | "
        f"已运行: {timer.get_elapsed()}"
    )

async def main():
    # 配置文件和路径
    progress_file = "progress_重复内容0814.txt"
    output_file = "output/processed_重复内容0814.xlsx"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 初始化计时器
    timer = Timer()
    timer.start()

    # 读取数据
    try:
        df = pd.read_excel("input/待抽取词条-汇总_test60.xlsx")
        rows = df.to_dict(orient="records")
        total_count = len(rows)
        logger.info(f"📊 共读取 {total_count} 条数据")
    except Exception as e:
        logger.error(f"读取CSV文件失败: {str(e)}")
        return

    # 加载处理进度
    start_index = load_progress(progress_file)
    logger.info(f"⏩ 从索引 {start_index} 开始处理（跳过前 {start_index} 条）")

    # 创建API客户端
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

    # 创建处理任务
    processed_count = 0
    tasks = []

    try:
        for idx in range(start_index, len(rows)):
            row = rows[idx]
            text = row.get("关键词所在章节对应的正文", "")

            if pd.isna(text) or not text.strip():
                logger.debug(f"⏭ 跳过空文本条目 {idx}")
                empty_record = create_empty_record(row, idx, "跳过: 空文本")
                save_partial_results([empty_record], output_file)
                save_progress(progress_file, idx + 1)
                processed_count += 1
                continue

            task = asyncio.create_task(process_text(client, row, idx))
            tasks.append(task)

            # 每批处理5条
            if len(tasks) >= 30:
                await process_batch(tasks, output_file, progress_file, idx + 1)
                processed_count += len(tasks)
                tasks = []
                log_progress(processed_count, total_count, timer)

        # 处理剩余任务
        if tasks:
            await process_batch(tasks, output_file, progress_file, len(rows))
            processed_count += len(tasks)
            log_progress(processed_count, total_count, timer)

        # 只有完全成功时才删除进度文件
        if os.path.exists(progress_file):
            os.remove(progress_file)
            logger.info("✅ 处理完成，已清理进度文件")

    except asyncio.CancelledError:
        logger.warning("🛑 任务被取消，保留进度文件以便恢复")
    except Exception as e:
        logger.error(f"‼️ 处理中断: {str(e)}，保留进度文件以便恢复")
    finally:
        timer.stop()



if __name__ == "__main__":
    asyncio.run(main())