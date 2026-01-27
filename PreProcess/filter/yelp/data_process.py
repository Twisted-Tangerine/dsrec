from __future__ import annotations
import os
import json
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Path constants (relative to project PreProcess root)
# ----------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(BASE_DIR, "raw_data", "yelp")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "yelp")

# ----------------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------------

def _ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# Data loading & preprocessing
# ----------------------------------------------------------------------------

def load_yelp_reviews(date_min: str, date_max: str, rating_score: float = 0.0) -> List[Tuple[str, str, int]]:
    """Load Yelp review json lines → (user, item, YYYYMMDDhhmmss int)."""
    review_path = os.path.join(RAW_DIR, "yelp_academic_dataset_review.json")
    datas: List[Tuple[str, str, int]] = []
    with open(review_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="reading reviews"):
            review = json.loads(line)
            user = review.get("user_id")
            item = review.get("business_id")
            rating = review.get("stars")
            date_str = review.get("date")
            if not (user and item and date_str):
                continue
            if date_str < date_min or date_str > date_max:
                continue
            if rating is not None and float(rating) <= rating_score:
                continue
            ts_int = int(date_str.replace("-", "").replace(":", "").replace(" ", ""))
            datas.append((str(user), str(item), ts_int))
    return datas


def load_yelp_business_meta(data_maps: Dict[str, Dict[str, str]]) -> Dict[str, dict]:
    meta_path = os.path.join(RAW_DIR, "yelp_academic_dataset_business.json")
    item_ids = set(data_maps["item2id"].keys())
    meta_infos: Dict[str, dict] = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="reading business meta"):
            info = json.loads(line)
            iid = info.get("business_id")
            if iid in item_ids:
                meta_infos[iid] = info
    return meta_infos

# ----------------------------------------------------------------------------
# Common utilities
# ----------------------------------------------------------------------------

def filter_common(datas: List[Tuple[str, str, int]], user_t: int, item_t: int):
    user_cnt, item_cnt = defaultdict(int), defaultdict(int)
    for u, i, _ in datas:
        user_cnt[u] += 1
        item_cnt[i] += 1

    user_hist: Dict[str, List[Tuple[str, int]]] = {}
    for u, i, ts in datas:
        if user_cnt[u] < user_t or item_cnt[i] < item_t:
            continue
        user_hist.setdefault(u, []).append((i, ts))

    user_seq: Dict[str, List[str]] = {}
    for u, item_ts in user_hist.items():
        item_ts.sort(key=lambda x: x[1])
        user_seq[u] = [i for i, _ in item_ts]
    return user_seq


def kcore_filter_interactions(datas: List[Tuple[str, str, int]], user_core: int, item_core: int):
    from collections import Counter
    interactions = list(datas)
    while True:
        u_cnt = Counter(u for u, _, _ in interactions)
        i_cnt = Counter(i for _, i, _ in interactions)
        drop_u = {u for u, c in u_cnt.items() if c < user_core}
        drop_i = {i for i, c in i_cnt.items() if c < item_core}
        if not drop_u and not drop_i:
            break
        interactions = [(u, i, t) for (u, i, t) in interactions if u not in drop_u and i not in drop_i]
    seq: Dict[str, List[Tuple[str, int]]] = {}
    for u, i, t in interactions:
        seq.setdefault(u, []).append((i, t))
    for u in list(seq.keys()):
        seq[u].sort(key=lambda x: x[1])
        seq[u] = [i for i, _ in seq[u]]
    return seq


def id_map(user_items: Dict[str, List[str]]):
    user2id, item2id, id2user, id2item = {}, {}, {}, {}
    uid_next = iid_next = 1
    mapped: Dict[str, List[str]] = {}
    for u, items in user_items.items():
        if u not in user2id:
            user2id[u] = str(uid_next)
            id2user[str(uid_next)] = u
            uid_next += 1
        new_items: List[str] = []
        for it in items:
            if it not in item2id:
                item2id[it] = str(iid_next)
                id2item[str(iid_next)] = it
                iid_next += 1
            new_items.append(item2id[it])
        mapped[user2id[u]] = new_items
    return mapped, uid_next - 1, iid_next - 1, {
        "user2id": user2id,
        "item2id": item2id,
        "id2user": id2user,
        "id2item": id2item,
    }


def get_attribute_yelp(meta_infos: Dict[str, dict], data_maps: dict, attribute_core: int):
    attr_freq = defaultdict(int)
    for info in meta_infos.values():
        cat_field = info.get("categories") or ""
        for c in [x.strip() for x in cat_field.split(",") if x.strip()]:
            attr_freq[c] += 1
    attr2id, id2attr, next_id = {}, {}, 1
    items2attrs, lens = {}, []
    for iid, info in meta_infos.items():
        kept: List[int] = []
        cat_field = info.get("categories") or ""
        for c in [x.strip() for x in cat_field.split(",") if x.strip()]:
            if attr_freq[c] >= attribute_core:
                if c not in attr2id:
                    attr2id[c] = next_id
                    id2attr[next_id] = c
                    next_id += 1
                kept.append(attr2id[c])
        items2attrs[data_maps["item2id"][iid]] = kept
        lens.append(len(kept))
    data_maps.update({"attribute2id": attr2id, "id2attribute": id2attr})
    return len(attr2id), (float(np.mean(lens)) if lens else 0.0), data_maps, items2attrs

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def process_yelp_data(
    user_core: int = 3,
    item_core: int = 3,
    attribute_core: int = 0,
    rating_score: float = 0.0,
    use_iterative_kcore: bool = True,  # default to K-core (LLMEmb style)
):
    np.random.seed(12345)
    date_min, date_max = "2000-01-01 00:00:00", "2099-12-31 23:59:59"

    reviews = load_yelp_reviews(date_min, date_max, rating_score)
    print(f"Loaded {len(reviews):,} raw reviews (after score filter)")

    if use_iterative_kcore:
        user_items = kcore_filter_interactions(reviews, user_core, item_core)
        print("Iterative K-core filtering applied")
    else:
        user_items = filter_common(reviews, user_core, item_core)
        print("One-pass common filter applied")

    user_items, user_num, item_num, data_maps = id_map(user_items)

    user_lens = [len(v) for v in user_items.values()]
    interact_num = int(np.sum(user_lens))
    sparsity = (1 - interact_num / (user_num * item_num)) * 100 if user_num and item_num else 0

    print(f"Users: {user_num}, Items: {item_num}, Interactions: {interact_num}, Sparsity: {sparsity:.2f}%")

    meta_infos = load_yelp_business_meta(data_maps)
    get_attribute_yelp(meta_infos, data_maps, attribute_core)

    _ensure_output_dir()
    seq_file = os.path.join(OUTPUT_DIR, "inter_seq.txt")
    with open(seq_file, "w", encoding="utf-8") as f:
        for u, items in user_items.items():
            f.write(u + " " + " ".join(items) + "\n")
    with open(os.path.join(OUTPUT_DIR, "id_map.json"), "w", encoding="utf-8") as f:
        json.dump(data_maps, f)
    with open(os.path.join(OUTPUT_DIR, "item2attributes.json"), "w", encoding="utf-8") as f:
        json.dump(meta_infos, f)

    convert_seq_to_inter(OUTPUT_DIR, seq_file)

    print("✅ Processing complete →", OUTPUT_DIR)


def convert_seq_to_inter(out_dir: str, seq_path: str):
    inter_path = os.path.join(out_dir, "inter.txt")
    if not os.path.exists(seq_path):
        print("[WARN] seq file missing", seq_path)
        return
    with open(seq_path, "r", encoding="utf-8") as f_in, open(inter_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            uid, *items = parts
            for it in items:
                f_out.write(f"{int(uid)} {int(it)}\n")
    try:
        os.remove(seq_path)
    except OSError:
        pass
    print("inter.txt generated and inter_seq.txt removed")

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser("Yelp preprocessing")
    p.add_argument("--user_core", type=int, default=5)
    p.add_argument("--item_core", type=int, default=5)
    p.add_argument("--rating_score", type=float, default=0.0)
    p.add_argument("--attribute_core", type=int, default=0)
    p.add_argument("--no_iterative_kcore", action="store_false", dest="use_iterative_kcore", help="Disable iterative K-core and use common filter instead")
    # by default flag not provided = True, providing flag sets it False
    args = p.parse_args()

    process_yelp_data(
        user_core=args.user_core,
        item_core=args.item_core,
        attribute_core=args.attribute_core,
        rating_score=args.rating_score,
        use_iterative_kcore=args.use_iterative_kcore,
    )