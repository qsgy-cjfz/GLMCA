# -*- coding: utf-8
"""
LLM活动中心坐标推断：直接将用户签到坐标+时间模式喂给LLM
让LLM输出活动中心坐标，替代MGM的距离聚类中心发现

用法:
  python llm_center_coordinate_builder.py --dataset Gowalla --sample   # 查看示例
  python llm_center_coordinate_builder.py --dataset Gowalla            # 生成输入
  python llm_center_coordinate_builder.py --dataset Gowalla --call_api # 调用API
"""
import os
import json
import time
import math
import argparse
import jsonlines
import requests
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


def load_poi_coos(data_name):
    poi_file = f"Dataset/{data_name}/{data_name}_poi_coos.txt"
    poi_coos = {}
    for line in open(poi_file, 'r'):
        parts = line.strip().split()
        lid, lat, lng = int(parts[0]), float(parts[1]), float(parts[2])
        poi_coos[lid] = (lat, lng)
    print(f"加载POI坐标: {len(poi_coos)}个")
    return poi_coos


def build_user_trajectories(data_name):
    checkin_file = f"Dataset/{data_name}/{data_name}_checkins.txt"
    poi_coos = load_poi_coos(data_name)

    user_poi_freq = defaultdict(lambda: defaultdict(int))
    user_poi_slots = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for line in open(checkin_file, 'r'):
        parts = line.strip().split()
        uid, lid, ctime = int(parts[0]), int(parts[1]), float(parts[2])
        if lid not in poi_coos:
            continue

        t = time.gmtime(ctime)
        h, wday = t.tm_hour, t.tm_wday
        is_weekend = wday >= 5
        is_daytime = 8 <= h < 18

        if is_weekend and is_daytime:
            slot = "周末白天"
        elif is_weekend and not is_daytime:
            slot = "周末晚间"
        elif not is_weekend and is_daytime:
            slot = "工作日白天"
        else:
            slot = "工作日晚间"

        user_poi_freq[uid][lid] += 1
        user_poi_slots[uid][lid][slot] += 1

    user_data = {}
    for uid in user_poi_freq:
        pois = []
        for lid, freq in user_poi_freq[uid].items():
            lat, lng = poi_coos[lid]
            pois.append({"lid": lid, "lat": lat, "lng": lng, "freq": freq,
                         "slots": dict(user_poi_slots[uid][lid])})
        pois.sort(key=lambda x: -x["freq"])
        user_data[uid] = pois

    print(f"构建用户轨迹: {len(user_data)}个用户")
    return user_data


def get_system_prompt(data_name):
    if data_name == "Gowalla":
        dataset_desc = "Gowalla(2009-2010年)"
        cities = "San Francisco, New York, Austin, Chicago, Los Angeles, Seattle, Boston, Kansas City, Dallas 等美国城市"
    elif data_name == "Yelp":
        dataset_desc = "Yelp Open Dataset(2016年)"
        cities = "Phoenix, Las Vegas, Pittsburgh, Cleveland, Madison, Tampa 等美国城市"
    else:
        dataset_desc = f"{data_name} 数据集"
        cities = "美国主要城市"

    return f"""你是一个地理空间分析专家。给定一个用户在位置社交网络{dataset_desc}上的签到记录，按地理区域汇总展示。

你的任务：分析签到点的空间聚集模式，识别用户的主要活动中心坐标。

分析方法：
1. 数据来自{dataset_desc}，主要覆盖{cities}
2. 根据各区域的签到频次和时间分布，推断主要活动中心
3. 利用你对这些城市地理的知识，为每个活动中心给出代表性坐标

请输出3-5个活动中心，严格使用以下JSON格式：
{{"centers": [{{"lat": 37.785, "lng": -122.409}}, {{"lat": 30.269, "lng": -97.749}}]}}

注意：只输出JSON对象本身，不要包含markdown代码块标记，不要其他任何文字。"""



def _region_key(lat, lng):
    return (round(lat, 1), round(lng, 1))


def _region_label(lat, lng):
    lat_c, lng_c = round(lat, 1), round(lng, 1)
    if 30.2 <= lat_c <= 30.4 and -97.9 <= lng_c <= -97.6:
        return "Austin Downtown"
    elif 37.7 <= lat_c <= 37.9 and -122.5 <= lng_c <= -122.3:
        return "SF Downtown"
    elif 40.6 <= lat_c <= 40.8 and -74.1 <= lng_c <= -73.7:
        return "NYC Metro"
    elif 47.5 <= lat_c <= 47.7 and -122.4 <= lng_c <= -122.2:
        return "Seattle Downtown"
    elif 33.9 <= lat_c <= 34.2 and -118.5 <= lng_c <= -118.1:
        return "LA Central"
    elif 38.8 <= lat_c <= 39.1 and -94.9 <= lng_c <= -94.5:
        return "Kansas City"
    elif 32.7 <= lat_c <= 33.2 and -97.0 <= lng_c <= -96.6:
        return "Dallas"
    elif 42.3 <= lat_c <= 42.4 and -71.1 <= lng_c <= -70.9:
        return "Boston"
    else:
        return f"[{lat_c:.1f},{lng_c:.1f}]"

def get_user_prompt(uid, pois):
    total_freq = sum(p["freq"] for p in pois)

    region_freq = defaultdict(int)
    region_slots = defaultdict(lambda: defaultdict(int))
    region_coos = defaultdict(list)

    for p in pois:
        rk = _region_key(p["lat"], p["lng"])
        region_freq[rk] += p["freq"]
        for slot, cnt in p["slots"].items():
            region_slots[rk][slot] += cnt
        region_coos[rk].append((p["lat"], p["lng"]))

    sorted_regions = sorted(region_freq.items(), key=lambda x: -x[1])
    n_regions = len(sorted_regions)

    dist_parts = []
    for rk, freq in sorted_regions:
        pct = freq / total_freq * 100
        label = _region_label(rk[0], rk[1])
        dist_parts.append(f"{label}({pct:.0f}%)")

    prompt = f"用户UID:{uid}，共{total_freq}次签到，{n_regions}个活动区域。\n"
    prompt += f"区域分布: [{', '.join(dist_parts)}]\n\n"
    prompt += "各区域详情:\n"

    for rk, freq in sorted_regions:
        pct = freq / total_freq * 100
        label = _region_label(rk[0], rk[1])
        coos = region_coos[rk]
        avg_lat = sum(c[0] for c in coos) / len(coos)
        avg_lng = sum(c[1] for c in coos) / len(coos)

        slot_parts = []
        for slot in ["工作日白天", "工作日晚间", "周末白天", "周末晚间"]:
            cnt = region_slots[rk].get(slot, 0)
            if cnt > 0:
                spct = cnt / freq * 100
                slot_parts.append(f"{slot}:{cnt}({spct:.0f}%)")
        prompt += f"  【{label}】[{avg_lat:.4f}, {avg_lng:.4f}] 共{freq}次({pct:.0f}%): {', '.join(slot_parts)}\n"

    prompt += "\n请分析空间聚集模式，输出活动中心坐标（JSON格式）。"
    return prompt

def build_input(data_name, user_data):
    os.makedirs("batch_input", exist_ok=True)
    output_file = f"batch_input/{data_name.lower()}_center_coordinate_input.jsonl"

    input_content = []
    skipped = 0
    for uid, pois in sorted(user_data.items()):
        total_freq = sum(p["freq"] for p in pois)
        if total_freq < 10:
            skipped += 1
            continue

        message = [
            {"role": "system", "content": get_system_prompt(data_name)},
            {"role": "user", "content": get_user_prompt(uid, pois)}
        ]
        row = {
            "custom_id": str(uid),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "deepseek-chat", "messages": message, "max_tokens": 400}
        }
        input_content.append(row)

    with jsonlines.open(output_file, mode='w') as writer:
        for row in input_content:
            writer.write(row)

    print(f"生成LLM输入: {len(input_content)}个用户 (跳过{skipped}个) → {output_file}")


API_KEY = "sk-2874f38dbe9f4c908ca7f85b5d07ad47"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(requests.exceptions.RequestException)
)
def call_api(payload):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    resp = requests.post("https://api.deepseek.com/v1/chat/completions",
                         headers=headers, json=payload, timeout=(100, 300))
    resp.raise_for_status()
    return resp.json()


def process_line(line):
    try:
        data = json.loads(line.strip())
        api_resp = call_api(data["body"])
        return {"custom_id": data["custom_id"],
                "raw_response": api_resp["choices"][0]["message"]["content"]}
    except Exception as e:
        return {"custom_id": data.get("custom_id", "unknown"), "error": str(e)}


def call_api_batch(data_name):
    input_file = f"batch_input/{data_name.lower()}_center_coordinate_input.jsonl"
    os.makedirs("batch_output", exist_ok=True)
    output_file = f"batch_output/{data_name.lower()}_center_coordinate_output.jsonl"

    if os.path.exists(output_file):
        print(f"[完成] 输出已存在: {output_file}，删除后重试")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        lines = list(f)

    print(f"[开始] 调用API，共{len(lines)}个用户...")
    start = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=10, thread_name_prefix="api") as exe:
        futures = [exe.submit(process_line, l) for l in lines]
        for f in tqdm(as_completed(futures), total=len(lines), desc="LLM坐标推断"):
            results.append(f.result())
    """ThreadPoolExecutor(max_workers=10)：创建最多 10 个并发线程
    exe.submit(process_line, l)：将每行 JSONL 数据（一个用户的签到轨迹）提交给线程池，process_line 解析后调用 LLM API 获取活动中心坐标
    as_completed(futures)：按完成顺序收集结果（而非提交顺序），配合 tqdm 显示进度条
    f.result()：获取每个线程的返回值（LLM 响应或错误信息）
    """

    results.sort(key=lambda x: x["custom_id"])
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = sum(1 for r in results if "error" not in r)
    print(f"[完成] 用时{time.time()-start:.1f}秒，成功{ok}/{len(results)} → {output_file}")


def print_sample(data_name, user_data, n=2):
    print(f"\n{'='*60}")
    for uid, pois in sorted(user_data.items()):
        if sum(p["freq"] for p in pois) < 30:
            continue
        print(f"\n--- 用户 {uid} ({sum(p['freq'] for p in pois)}次签到) ---")
        print(get_user_prompt(uid, pois))
        n -= 1
        if n <= 0:
            break
def pretty_print(data_name, n=5):
    input_file = f"batch_input/{data_name.lower()}_center_coordinate_input.jsonl"
    if not os.path.exists(input_file):
        print(f"文件不存在: {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"共{len(lines)}条记录，展示前{n}条：\n")
    for i, line in enumerate(lines[:n]):
        data = json.loads(line)
        uid = data["custom_id"]
        sys_msg = data["body"]["messages"][0]["content"]
        user_msg = data["body"]["messages"][1]["content"]
        print(f"{'='*60}")
        print(f"用户 {uid}")
        print(f"{'='*60}")
        print(f"[System]: {sys_msg[:200]}...")
        print(f"\n[User]:")
        print(user_msg)
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Gowalla', type=str)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--call_api', action='store_true')
    parser.add_argument('--pretty', action='store_true', help='格式化打印前几条记录')
    args = parser.parse_args()

    user_data = build_user_trajectories(args.dataset)
    if args.pretty:
        pretty_print(args.dataset)
    elif args.sample:
        print_sample(args.dataset, user_data)
    elif args.call_api:
        call_api_batch(args.dataset)
    else:
        print_sample(args.dataset, user_data, n=1)
        build_input(args.dataset, user_data)
