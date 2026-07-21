# -*- coding: utf-8
"""
解析LLM输出的活动中心坐标
输出: processed/activity_centers/{dataset}_llm_coordinate_centers.json

用法: python parse_llm_coordinate_centers.py --dataset Gowalla
"""
import os
import json
import re
import argparse


def extract_json(text):
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        return json.loads(match.group())
    return None


def parse_centers(data_name):
    output_file = f"batch_output/{data_name.lower()}_center_coordinate_output.jsonl"
    result_file = f"processed/activity_centers/{data_name}_llm_coordinate_centers.json"

    user_centers = {}
    error_count = parse_fail = 0

    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            uid = data["custom_id"]
            if "error" in data:
                error_count += 1
                continue
            try:
                parsed = extract_json(data["raw_response"].strip())
                if not parsed or "centers" not in parsed:
                    parse_fail += 1
                    continue
                centers = []
                for c in parsed["centers"]:
                    lat, lng = float(c["lat"]), float(c["lng"])
                    if -90 <= lat <= 90 and -180 <= lng <= 180:
                        centers.append({"lat": lat, "lng": lng})
                if centers:
                    user_centers[uid] = centers
                else:
                    parse_fail += 1
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                parse_fail += 1

    os.makedirs("processed/activity_centers", exist_ok=True)
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({"user_centers": user_centers, "total_users": len(user_centers),
                    "error_count": error_count, "parse_fail_count": parse_fail},
                   f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"成功: {len(user_centers)} | API失败: {error_count} | 解析失败: {parse_fail}")
    print(f"[保存] {result_file}")
    for i, (uid, cs) in enumerate(user_centers.items()):
        if i >= 5: break
        coords = ", ".join(f"({c['lat']:.4f}, {c['lng']:.4f})" for c in cs)
        print(f"  用户{uid}: [{coords}]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Gowalla', type=str)
    args = parser.parse_args()
    parse_centers(args.dataset)
