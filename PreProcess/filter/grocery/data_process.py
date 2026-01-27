import os
import json
import time
from collections import defaultdict
from tqdm import tqdm
import numpy as np


def _normalize_timestamp(ts):
    """Best-effort normalize to sortable integer timestamp.
    Strategy:
    - If string date, try to parse to YYYYMMDD (not expected here).
    - If int milliseconds (>=1e12), divide to seconds.
    - If seconds are plausible (2000-01-01 ~ 2035-01-01), keep.
    - Else fallback to raw int (still sortable); caller can override with sequence index if needed.
    """
    try:
        if isinstance(ts, str):
            ts_int = int(ts)
        else:
            ts_int = int(ts)
    except Exception:
        return 0

    # ms -> s
    if ts_int >= 10**12:
        ts_int = ts_int // 1000

    # plausible seconds range
    if 946684800 <= ts_int <= 2051222400:
        return ts_int

    # try formatting even if out-of-range
    try:
        ymd = time.strftime('%Y%m%d', time.gmtime(ts_int))
        return int(ymd)
    except Exception:
        return ts_int


# Pre-defined directories (relative to the project `PreProcess` root)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DIR = os.path.join(BASE_DIR, 'raw_data', 'grocery')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'grocery')

def load_grocery_data(review_path=None, rating_score=0.0):
    """Read 2018-style Amazon reviews -> list of (user, item, time_YYYYMMDD).
    Keys: overall, reviewerID, asin, unixReviewTime/reviewTime
    """
    if review_path is None:
        review_path = os.path.join(RAW_DIR, 'Grocery_and_Gourmet_Food.jsonl')
    """Read 2018-style Amazon reviews -> list of (user, item, time_YYYYMMDD).
    Keys: overall, reviewerID, asin, unixReviewTime/reviewTime
    """
    datas = []
    with open(review_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='reading reviews'):
            obj = json.loads(line.strip())
            user = obj.get('reviewerID')
            item = obj.get('asin')
            rating_val = obj.get('overall')
            if not user or not item:
                continue
            # timestamp: prefer unixReviewTime; fallback to reviewTime (e.g., "07 23, 2014")
            ts_raw = obj.get('unixReviewTime')
            if ts_raw is None:
                rt = obj.get('reviewTime')
                try:
                    ts_norm = int(time.strftime('%Y%m%d', time.strptime(str(rt), '%m %d, %Y')))
                except Exception:
                    ts_norm = 0
            else:
                try:
                    ts_norm = int(time.strftime('%Y%m%d', time.gmtime(int(ts_raw))))
                except Exception:
                    ts_norm = _normalize_timestamp(ts_raw)
            try:
                rating = float(rating_val) if rating_val is not None else None
            except Exception:
                rating = None
            if rating is not None and rating <= rating_score:
                continue
            datas.append((str(user), str(item), int(ts_norm)))
    return datas


def load_grocery_meta_data(datamaps, meta_path=None):
    """Build meta dict (2018 schema) for items in id map. Key: asin."""
    meta_infos = {}
    item_ids = set(datamaps['item2id'].keys())
    if meta_path is None:
        meta_path = os.path.join(RAW_DIR, 'meta_Grocery_and_Gourmet_Food.jsonl')
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='reading meta'):
            info = json.loads(line.strip())
            key = info.get('asin')
            if not key or key not in item_ids:
                continue
            meta_infos[key] = info
    return meta_infos


import argparse  # keep for process CLI only


def add_comma(num):
    s = str(num)
    res = ''
    for i in range(len(s)):
        res += s[i]
        if (len(s)-i-1) % 3 == 0:
            res += ','
    return res[:-1]


def filter_common(user_items, user_t, item_t):
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, item, _ in user_items:
        user_count[user] += 1
        item_count[item] += 1

    User = {}
    for user, item, timestamp in user_items:
        if user_count[user] < user_t or item_count[item] < item_t:
            continue
        if user not in User:
            User[user] = []
        User[user].append((item, timestamp))

    new_User = {}
    for userid in User.keys():
        User[userid].sort(key=lambda x: x[1])
        new_hist = [i for i, t in User[userid]]
        new_User[userid] = new_hist

    return new_User


def kcore_filter_interactions(datas, user_core=5, item_core=5):
    """Iterative K-core filtering on raw interactions (u,i,t). Returns user->sorted item sequence dict."""
    from collections import Counter
    interactions = list(datas)
    while True:
        user_count = Counter(u for u, _, _ in interactions)
        item_count = Counter(i for _, i, _ in interactions)
        to_remove_users = {u for u, c in user_count.items() if c < user_core}
        to_remove_items = {i for i, c in item_count.items() if c < item_core}
        if not to_remove_users and not to_remove_items:
            break
        interactions = [(u, i, t) for (u, i, t) in interactions if u not in to_remove_users and i not in to_remove_items]
    user_seq = {}
    for u, i, t in interactions:
        if u not in user_seq:
            user_seq[u] = []
        user_seq[u].append((i, t))
    for u in list(user_seq.keys()):
        user_seq[u].sort(key=lambda x: x[1])
        user_seq[u] = [i for i, _ in user_seq[u]]
    return user_seq

def id_map(user_items):
    user2id, item2id = {}, {}
    id2user, id2item = {}, {}
    user_id = 1
    item_id = 1
    final_data = {}

    for user, items in user_items.items():
        if user not in user2id:
            user2id[user] = str(user_id)
            id2user[str(user_id)] = user
            user_id += 1
        iids = []
        for item in items:
            if item not in item2id:
                item2id[item] = str(item_id)
                id2item[str(item_id)] = item
                item_id += 1
            iids.append(item2id[item])
        final_data[user2id[user]] = iids

    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item
    }
    return final_data, user_id - 1, item_id - 1, data_maps


def get_counts(user_items):
    user_count = {}
    item_count = {}
    for user, items in user_items.items():
        user_count[user] = len(items)
        for item in items:
            if item not in item_count:
                item_count[item] = 1
            else:
                item_count[item] += 1
    return user_count, item_count


def _flatten_categories(val):
    if val is None:
        return []
    if isinstance(val, str):
        return [val.strip()] if val.strip() else []
    res = []
    if isinstance(val, list):
        for c in val:
            if isinstance(c, list):
                res.extend([str(x).strip() for x in c if str(x).strip()])
            else:
                s = str(c).strip()
                if s:
                    res.append(s)
    return res


def get_attribute_Grocery(meta_infos, datamaps, attribute_core=0):
    """Attribute extraction for 2018 schema: category + main_cat + brand + feature list."""
    attributes = defaultdict(int)
    for iid, info in meta_infos.items():
        cates = _flatten_categories(info.get('category'))
        main_cat = info.get('main_cat')
        brand = info.get('brand')
        feats = info.get('feature') if isinstance(info.get('feature'), list) else []
        for cate in cates:
            attributes[cate] += 1
        if main_cat:
            attributes[str(main_cat).strip()] += 1
        if brand:
            attributes[str(brand).strip()] += 1
        for ft in feats:
            if isinstance(ft, str) and ft.strip():
                attributes[ft.strip()] += 1

    new_meta = {}
    for iid, info in meta_infos.items():
        new_meta[iid] = []
        cates = _flatten_categories(info.get('category'))
        main_cat = info.get('main_cat')
        brand = info.get('brand')
        feats = info.get('feature') if isinstance(info.get('feature'), list) else []
        for cate in cates:
            if attributes[cate] >= attribute_core:
                new_meta[iid].append(cate)
        if main_cat:
            m = str(main_cat).strip()
            if attributes[m] >= attribute_core:
                new_meta[iid].append(m)
        if brand:
            b = str(brand).strip()
            if attributes[b] >= attribute_core:
                new_meta[iid].append(b)
        for ft in feats:
            if isinstance(ft, str) and ft.strip() and attributes[ft.strip()] >= attribute_core:
                new_meta[iid].append(ft.strip())

    attribute2id = {}
    id2attribute = {}
    attribute_id = 1
    items2attributes = {}
    attribute_lens = []
    for iid, attrs in new_meta.items():
        item_id = datamaps['item2id'][iid]
        items2attributes[item_id] = []
        for attribute in attrs:
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[attribute_id] = attribute
                attribute_id += 1
            items2attributes[item_id].append(attribute2id[attribute])
        attribute_lens.append(len(items2attributes[item_id]))

    print(f'after delete, attribute num:{len(attribute2id)}')
    if attribute_lens:
        print(f'attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, Avg.:{np.mean(attribute_lens):.4f}')

    datamaps['attribute2id'] = attribute2id
    datamaps['id2attribute'] = id2attribute
    return len(attribute2id), (float(np.mean(attribute_lens)) if attribute_lens else 0.0), datamaps, items2attributes


def process_grocery_data(user_core=5, item_core=5, attribute_core=0, rating_score=0.0):
    np.random.seed(12345)

    datas = load_grocery_data(rating_score=rating_score)
    print(f'Grocery Raw data has been processed! Ratings <= {rating_score} are deleted!')

    # # one-pass common filter (align with trip/yelp filter_common)
    # user_items = filter_common(datas, user_t=user_core, item_t=item_core)
    # print(f'User {user_core}-core complete! Item {item_core}-core complete!')

    # strict iterative K-core filtering
    user_items = kcore_filter_interactions(datas, user_core=user_core, item_core=item_core)
    print(f'K-core filtering complete! user_core={user_core}, item_core={item_core}')

    user_items, user_num, item_num, data_maps = id_map(user_items)

    user_count, item_count = get_counts(user_items)
    user_count_list = list(user_count.values())
    item_count_list = list(item_count.values())
    user_avg, user_min, user_max = (np.mean(user_count_list) if user_count_list else 0), (np.min(user_count_list) if user_count_list else 0), (np.max(user_count_list) if user_count_list else 0)
    item_avg, item_min, item_max = (np.mean(item_count_list) if item_count_list else 0), (np.min(item_count_list) if item_count_list else 0), (np.max(item_count_list) if item_count_list else 0)
    interact_num = int(np.sum([x for x in user_count_list]))
    sparsity = (1 - interact_num / (max(1, user_num) * max(1, item_num))) * 100

    show_info = f'Total User: {user_num}, Avg User: {user_avg:.4f}, Min Len: {user_min}, Max Len: {user_max}\n' + \
                f'Total Item: {item_num}, Avg Item: {item_avg:.4f}, Min Inter: {item_min}, Max Inter: {item_max}\n' + \
                f'Iteraction Num: {interact_num}, Sparsity: {sparsity:.2f}%'
    print(show_info)

    print('Begin extracting meta infos...')
    meta_infos_original = load_grocery_meta_data(data_maps)

    print('Begin processing attributes...')
    attribute_num, avg_attribute, data_maps, item2attributes = get_attribute_Grocery(
        meta_infos_original, data_maps, attribute_core
    )

    print(f'Grocery & {add_comma(user_num)}& {add_comma(item_num)} & {user_avg:.1f}'
          f'& {item_avg:.1f}& {add_comma(interact_num)}& {sparsity:.2f}\%&{add_comma(attribute_num)}&'
          f'{avg_attribute:.1f} \\')
    print("--------------------------")

    handled_path = OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data_file = os.path.join(handled_path, 'inter_seq.txt')
    id_file = os.path.join(handled_path, 'id_map.json')
    meta_file = os.path.join(handled_path, 'item2attributes.json')

    print(f"Writing processed data to {OUTPUT_DIR}...")

    with open(data_file, 'w', encoding='utf-8') as out:
        for user, items in user_items.items():
            out.write(user + ' ' + ' '.join(items) + '\n')

    with open(id_file, 'w', encoding='utf-8') as f:
        json.dump(data_maps, f)

    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(meta_infos_original, f)

    print(f"✅ Processed Grocery dataset saved to {handled_path}")
    return handled_path, data_file


def convert_seq_to_inter(handled_path, seq_file_path):
    inter_file_path = os.path.join(handled_path, 'inter.txt')

    print(f"\nReading sequential data from {seq_file_path} for conversion...")
    data = {}

    if not os.path.exists(seq_file_path):
        print(f"Error: Input file not found at {seq_file_path}")
        return

    with open(seq_file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line_data = line.rstrip().split(' ')
            if not line_data or len(line_data) < 2:
                continue
            user_id = line_data[0]
            line_data.pop(0)
            data[user_id] = line_data

    print(f"Writing flat interaction data to {inter_file_path}...")
    with open(inter_file_path, 'w', encoding='utf-8') as f:
        for user, item_list in tqdm(data.items()):
            for item in item_list:
                try:
                    u = int(user)
                    i = int(item)
                    f.write('%s %s\n' % (u, i))
                except ValueError:
                    print(f"Skipping invalid data: user='{user}', item='{item}'")

    print(f"✅ Final interaction file 'inter.txt' saved to {handled_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grocery dataset processing (2018 schema)')
    parser.add_argument('--user_core', type=int, default=8)
    parser.add_argument('--item_core', type=int, default=8)
    parser.add_argument('--rating_score', type=float, default=0.0)
    parser.add_argument('--attribute_core', type=int, default=0)
    parser.add_argument('--keep_seq', action='store_true', help='Keep intermediate inter_seq.txt file')
    # 2018 schema fixed: asin only
    args = parser.parse_args()

    output_path, seq_file = process_grocery_data(
        user_core=args.user_core,
        item_core=args.item_core,
        attribute_core=args.attribute_core,
        rating_score=args.rating_score
    )

    convert_seq_to_inter(output_path, seq_file)

    # remove intermediate seq file unless requested to keep
    if not args.keep_seq and os.path.exists(seq_file):
        try:
            os.remove(seq_file)
        except OSError:
            pass

    print("\nAll Grocery processing steps complete.")


