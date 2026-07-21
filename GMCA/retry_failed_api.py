# -*- coding: utf-8
"""
重试失败的API调用
读取batch_output中error的记录，重新调用API，合并回原文件

用法:
  python retry_failed_api.py --dataset Gowalla
  python retry_failed_api.py --dataset Yelp
  python retry_failed_api.py --dataset Gowalla --rounds 3   # 多轮重试
"""
import os
import json
import time
import argparse
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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


def retry_dataset(data_name, max_rounds=1):
    input_file = f"batch_input/{data_name.lower()}_center_coordinate_input.jsonl"
    output_file = f"batch_output/{data_name.lower()}_center_coordinate_output.jsonl"

    if not os.path.exists(output_file):
        print(f"输出文件不存在: {output_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        input_map = {}
        for line in f:
            data = json.loads(line.strip())
            input_map[data["custom_id"]] = data

    with open(output_file, "r", encoding="utf-8") as f:
        results = {}
        for line in f:
            data = json.loads(line.strip())
            results[data["custom_id"]] = data

    for round_idx in range(max_rounds):
        failed = {cid: r for cid, r in results.items() if "error" in r}
        if not failed:
            print("无失败记录，无需重试")
            break

        print(f"\n[第{round_idx+1}轮重试] 共{len(failed)}个失败用户...")
        start = time.time()

        failed_inputs = [input_map[cid] for cid in failed if cid in input_map]
        missing = [cid for cid in failed if cid not in input_map]
        if missing:
            print(f"  警告: {len(missing)}个失败用户在输入文件中找不到，跳过")

        new_results = []
        with ThreadPoolExecutor(max_workers=10, thread_name_prefix="retry") as exe:
            def process(inp):
                try:
                    api_resp = call_api(inp["body"])
                    return {"custom_id": inp["custom_id"],
                            "raw_response": api_resp["choices"][0]["message"]["content"]}
                except Exception as e:
                    return {"custom_id": inp["custom_id"], "error": str(e)}

            futures = [exe.submit(process, inp) for inp in failed_inputs]
            for f in tqdm(as_completed(futures), total=len(failed_inputs), desc=f"重试第{round_idx+1}轮"):
                new_results.append(f.result())

        recovered = 0
        for r in new_results:
            cid = r["custom_id"]
            if "error" not in r:
                recovered += 1
            results[cid] = r

        elapsed = time.time() - start
        still_failed = sum(1 for r in results.values() if "error" in r)
        print(f"  本轮恢复{recovered}个，仍失败{still_failed}个，用时{elapsed:.1f}秒")

    sorted_results = sorted(results.values(), key=lambda x: int(x["custom_id"]) if x["custom_id"].isdigit() else x["custom_id"])
    with open(output_file, "w", encoding="utf-8") as f:
        for r in sorted_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(results)
    ok = sum(1 for r in results.values() if "error" not in r)
    print(f"\n[完成] 总计{total}个，成功{ok}，失败{total-ok} → {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Gowalla', type=str)
    parser.add_argument('--rounds', default=1, type=int, help='重试轮数')
    args = parser.parse_args()
    retry_dataset(args.dataset, args.rounds)
