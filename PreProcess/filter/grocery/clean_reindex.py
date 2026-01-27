"""clean_reindex.py
Two-stage cleaning & re-indexing for Grocery dataset.
Stage-1  删除原始缺描述項 `MISSING_ITEMS` ，生成第一轮连续索引 temp_idx。
Stage-2  再删除 bad 向量項 `BAD_ITEMS`（基于 temp_idx），并生成最终连续索引 final_idx。
最终覆盖:
    output/grocery/inter.txt
    output/grocery/id_map.json
"""
import json, os
from collections import OrderedDict

# ---------- 路径 ----------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(BASE_DIR, "output", "grocery")
INTER_PATH = os.path.join(OUT_DIR, "inter.txt")
IDMAP_PATH = os.path.join(OUT_DIR, "id_map.json")

# ---------- 待删除集合 ----------
MISSING_ITEMS = {
    275, 1345, 2743, 4151, 5014, 5030, 5056, 7950,
    8032, 8666, 9388, 9993, 10266, 12525, 13552, 15043
}
BAD_ITEMS = {904, 5311, 5512, 6374, 11781, 12838, 14000, 14266, 15101}


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------

def load_interactions(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u, i = line.split()[:2]
            yield int(u), int(i)


def write_interactions(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for u, i in lines:
            f.write(f"{u} {i}\n")


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main():
    if not os.path.exists(INTER_PATH) or not os.path.exists(IDMAP_PATH):
        raise FileNotFoundError("inter.txt 或 id_map.json 缺失，请先运行前置脚本")

    id_map = json.load(open(IDMAP_PATH, "r", encoding="utf-8"))
    id2item_orig = id_map["id2item"]  # key str(idx) -> asin

    # ---------------- Stage-1: 删除 MISSING_ITEMS ----------------
    stage1_lines = []
    stage1_item_order = OrderedDict()
    for u, old_i in load_interactions(INTER_PATH):
        if old_i in MISSING_ITEMS:
            continue
        stage1_lines.append((u, old_i))
        if old_i not in stage1_item_order:
            stage1_item_order[old_i] = None

    # temp mapping old -> temp_idx
    old2temp = {old: idx for idx, old in enumerate(stage1_item_order.keys(), start=1)}

    # translate to temp idx
    stage1_lines_translated = [(u, old2temp[old_i]) for u, old_i in stage1_lines]

    # ---------------- Stage-2: 删除 BAD_ITEMS (基于 temp idx) ------
    stage2_lines = []
    stage2_item_order = OrderedDict()
    removed_bad = 0
    for u, temp_i in stage1_lines_translated:
        if temp_i in BAD_ITEMS:
            removed_bad += 1
            continue
        stage2_lines.append((u, temp_i))
        if temp_i not in stage2_item_order:
            stage2_item_order[temp_i] = None

    # final mapping temp -> final_idx (连续)
    temp2final = {temp: idx for idx, temp in enumerate(stage2_item_order.keys(), start=1)}

    # translate to final idx
    final_lines = [(u, temp2final[temp_i]) for u, temp_i in stage2_lines]

    print(f"Stage-1 kept {len(stage1_item_order)} items; Stage-2 removed {removed_bad} bad items; Final items {len(temp2final)}")

    # ---------------- 写 inter.txt (覆盖) ----------------
    write_interactions(INTER_PATH, final_lines)
    print("inter.txt 已覆盖")

    # ---------------- 重建 id_map.json ----------------
    item2id_new = {}
    id2item_new = {}
    for old_i, temp_idx in old2temp.items():
        if temp_idx not in temp2final:
            continue  # 被 bad 删除
        final_idx = temp2final[temp_idx]
        asin = id2item_orig.get(str(old_i), "unknown")
        item2id_new[asin] = str(final_idx)
        id2item_new[str(final_idx)] = asin

    id_map["item2id"] = item2id_new
    id_map["id2item"] = id2item_new
    with open(IDMAP_PATH, "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False)
    print("id_map.json 已重建 (连续索引)")


if __name__ == "__main__":
    main()
