import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import datetime
import ast

# Pre-defined directories (relative to the project `PreProcess` root)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DIR = os.path.join(BASE_DIR, 'raw_data', 'trip')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'trip')


def load_trip_data():
    """读取 TripAdvisor 的 reviews 与 offering 数据"""
    review_path = os.path.join(RAW_DIR, 'reviews.csv')
    offering_path = os.path.join(RAW_DIR, 'offerings.csv')
    print(f"Loading TripAdvisor reviews from {review_path}")

    reviews = pd.read_csv(review_path)
    offerings = pd.read_csv(offering_path)
    print(f"Reviews: {len(reviews):,} rows | Offerings: {len(offerings):,} rows")
    
    return reviews, offerings


def preprocess_trip(reviews, rating_score=0.0):
    """将 TripAdvisor 的 reviews.csv 转换为 (user, item, time) 格式"""
    inter_data = []

    for _, row in tqdm(reviews.iterrows(), total=len(reviews)):
        try:
            author_info = json.loads(row['author'].replace("'", '"'))
            user = author_info.get('username', None)
            if not user:
                continue

            item = str(int(row['offering_id']))  # offering_id
            date_str = row['date']
            
            # 解析时间为 YYYYMMDD 格式，与Yelp保持一致
            try:
                dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                timestamp = int(dt.strftime("%Y%m%d"))
            except:
                continue

            ratings = json.loads(row['ratings'].replace("'", '"'))
            overall = ratings.get('overall', None)
            if overall is None or float(overall) <= rating_score:
                continue

            # 只包含(user, item, time)，不包含rating
            inter_data.append((user, item, timestamp))
        except Exception as e:
            continue

    print(f"✅ Parsed {len(inter_data):,} valid interactions.")
    return inter_data

def get_interaction(datas):
    """将 (user, item, time) 转为按时间排序的序列"""
    user_seq = {}
    for (user, item, time) in datas:
        if user in user_seq:
            user_seq[user].append((item, time))
        else:
            user_seq[user] = []
            user_seq[user].append((item, time))
    
    for user, item_time in user_seq.items():
        item_time.sort(key=lambda x: x[1])  # 对各个数据集得单独排序
        items = []
        for t in item_time:
            items.append(t[0])
        user_seq[user] = items
    return user_seq


def filter_common(datas, user_t=3, item_t=3):
    """ K-core 筛选 """
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for (user, item, _) in datas:
        user_count[user] += 1
        item_count[item] += 1

    User = {}
    for (user, item, timestamp) in datas:
        if user_count[user] < user_t or item_count[item] < item_t:
            continue
        if user not in User.keys():
            User[user] = []
        User[user].append((item, timestamp))

    new_User = {}
    for userid in User.keys():
        User[userid].sort(key=lambda x: x[1])
        new_hist = [i for i, t in User[userid]]
        new_User[userid] = new_hist

    return new_User


def id_map(user_items):
    """建立 user2id / item2id"""
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


def load_trip_meta_data(datamaps, offerings):
    """提取 TripAdvisor offering 元信息，包含更多有用字段"""
    meta_infos = {}
    item_ids = list(datamaps['item2id'].keys())
    for _, row in tqdm(offerings.iterrows(), total=len(offerings)):
        oid = str(int(row['id']))
        if oid not in item_ids:
            continue
        
        # 构建更丰富的categories字段
        categories = row.get('type', '')
        if row.get('hotel_class') and not pd.isna(row.get('hotel_class')):
            categories += f", {row.get('hotel_class')} star"
        
        # 处理地址信息
        address_info = row.get('address', '')
        if isinstance(address_info, str) and address_info.startswith('{'):
            try:
                address_dict = ast.literal_eval(address_info)
                address_info = address_dict
            except:
                pass
        
        meta_infos[oid] = {
            "name": row.get('name', ''),
            "type": row.get('type', ''),
            "categories": categories,
            "address": address_info,
            "url": row.get('url', ''),
            "phone": row.get('phone', ''),
            "details": row.get('details', ''),
            "hotel_class": row.get('hotel_class', None),
            "region_id": row.get('region_id', None)
        }
    return meta_infos


def get_attribute_Trip(meta_infos, datamaps, attribute_core):
    """属性处理"""
    attributes = defaultdict(int)
    for iid, info in tqdm(meta_infos.items()):
        try:
            cates = [cate.strip() for cate in info['categories'].split(',')]
            for cate in cates:
                attributes[cate] += 1
        except:
            pass
    print(f'before delete, attribute num:{len(attributes)}')
    new_meta = {}
    for iid, info in tqdm(meta_infos.items()):
        new_meta[iid] = []
        try:
            cates = [cate.strip() for cate in info['categories'].split(',')]
            for cate in cates:
                if attributes[cate] >= attribute_core:
                    new_meta[iid].append(cate)
        except:
            pass
    # 做映射
    attribute2id = {}
    id2attribute = {}
    attribute_id = 1
    items2attributes = {}
    attribute_lens = []
    # load id map
    for iid, attributes in new_meta.items():
        item_id = datamaps['item2id'][iid]
        items2attributes[item_id] = []
        for attribute in attributes:
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[attribute_id] = attribute
                attribute_id += 1
            items2attributes[item_id].append(attribute2id[attribute])
        attribute_lens.append(len(items2attributes[item_id]))
    print(f'after delete, attribute num:{len(attribute2id)}')
    print(f'attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, Avg.:{np.mean(attribute_lens):.4f}')
    # 更新datamap
    datamaps['attribute2id'] = attribute2id
    datamaps['id2attribute'] = id2attribute
    return len(attribute2id), np.mean(attribute_lens), datamaps, items2attributes


def process_trip_data(user_core=3, item_core=3, attribute_core=0, rating_score=0.0):
    """主处理函数：加载、处理并保存序列数据和元数据。"""
    np.random.seed(12345)
    
    reviews, offerings = load_trip_data()
    datas = preprocess_trip(reviews, rating_score=rating_score)
    
    print(f'TripAdvisor Raw data has been processed! Ratings <= {rating_score} are deleted!')
    user_items = filter_common(datas, user_t=user_core, item_t=item_core)
    print(f'User {user_core}-core complete! Item {item_core}-core complete!')
    
    user_items, user_num, item_num, data_maps = id_map(user_items)
    
    # 统计信息输出
    user_count = {}
    item_count = {}
    for user, items in user_items.items():
        user_count[user] = len(items)
        for item in items:
            if item not in item_count.keys():
                item_count[item] = 1
            else:
                item_count[item] += 1
    
    user_count_list = list(user_count.values())
    user_avg, user_min, user_max = np.mean(user_count_list), np.min(user_count_list), np.max(user_count_list)
    item_count_list = list(item_count.values())
    item_avg, item_min, item_max = np.mean(item_count_list), np.min(item_count_list), np.max(item_count_list)
    interact_num = np.sum([x for x in user_count_list])
    sparsity = (1 - interact_num / (user_num * item_num)) * 100
    
    show_info = f'Total User: {user_num}, Avg User Len: {user_avg:.4f}, Min Len: {user_min}, Max Len: {user_max}\n' + \
                f'Total Item: {item_num}, Avg Item Inter: {item_avg:.4f}, Min Inter: {item_min}, Max Inter: {item_max}\n' + \
                f'Iteraction Num: {interact_num}, Sparsity: {sparsity:.2f}%'
    print(show_info)

    print('Begin extracting meta infos...')
    meta_infos_original = load_trip_meta_data(data_maps, offerings) # 这是原始元数据
    
    print('Begin processing attributes...')
    attribute_num, avg_attribute, data_maps, item2attributes = get_attribute_Trip(
        meta_infos_original, data_maps, attribute_core
    )
    
    # 统计信息输出
    print(f'TripAdvisor & {user_num:,}& {item_num:,} & {user_avg:.1f}'
          f'& {item_avg:.1f}& {interact_num:,}& {sparsity:.2f}%&{attribute_num:,}&'
          f'{avg_attribute:.1f} \\')
    print("--------------------------")

    # --- 输出 ---
    handled_path = OUTPUT_DIR
    os.makedirs(handled_path, exist_ok=True)
    
    data_file = os.path.join(handled_path, 'inter_seq.txt')
    id_file = os.path.join(handled_path, 'id_map.json')
    meta_file = os.path.join(handled_path, 'item2attributes.json') 

    print(f"Writing processed data to {handled_path}...")
    
    # 保存 inter_seq.txt
    with open(data_file, 'w') as f:
        for user, items in user_items.items():
            f.write(user + ' ' + ' '.join(items) + '\n')
    
    # 保存 id_map.json
    with open(id_file, 'w') as f:
        json.dump(data_maps, f)
            
    # 保存 item2attributes.json
    with open(meta_file, 'w') as f:
        json.dump(meta_infos_original, f)

    print(f"✅ Processed TripAdvisor dataset (v2 logic) saved to {handled_path}")
    # 返回创建的路径，供下一步使用
    return handled_path, data_file


def convert_seq_to_inter(handled_path, seq_file_path):
    """将序列文件 (inter_seq.txt) 转换为扁平的交互文件 (inter.txt)"""
    
    # 定义输出文件路径
    inter_file_path = os.path.join(handled_path, 'inter.txt')

    print(f"\nReading sequential data from {seq_file_path} for conversion...")
    data = {}
    
    # 检查输入文件是否存在
    if not os.path.exists(seq_file_path):
        print(f"Error: Input file not found at {seq_file_path}")
        return

    # 读取 inter_seq.txt
    with open(seq_file_path, 'r') as f:
        for line in tqdm(f):
            line_data = line.rstrip().split(' ')
            if not line_data or len(line_data) < 2: 
                continue
            user_id = line_data[0]
            line_data.pop(0)    # delete user_id
            data[user_id] = line_data

    print(f"Writing flat interaction data to {inter_file_path}...")
    # 写入 inter.txt
    with open(inter_file_path, 'w') as f:
        for user, item_list in tqdm(data.items()):
            for item in item_list:
                try:
                    u = int(user)
                    i = int(item)
                    f.write('%s %s\n' % (u, i))
                except ValueError:
                    print(f"Skipping invalid data: user='{user}', item='{item}'")
    
    print(f"✅ Final interaction file 'inter.txt' saved to {handled_path}")

    # 默认删除中间 seq 文件（保持与 grocery 流程一致）
    if os.path.exists(seq_file_path):
        try:
            os.remove(seq_file_path)
        except OSError:
            pass



if __name__ == "__main__":

    USER_CORE = 3
    ITEM_CORE = 3
    RATING_SCORE = 0.0
    ATTRIBUTE_CORE = 0 
    
    output_path, seq_file = process_trip_data(
        user_core=USER_CORE, 
        item_core=ITEM_CORE, 
        attribute_core=ATTRIBUTE_CORE,
        rating_score=RATING_SCORE
    )
    
    convert_seq_to_inter(output_path, seq_file)
    
    print("\nAll processing steps complete.")