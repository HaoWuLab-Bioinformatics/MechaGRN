import argparse

import torch
from Model import MechaGRN
import pandas as pd
import numpy as np
from preprocessing import scRNADataset, load_data, adj2saprse_tensor, Feature_discretization_data
import os
import random
import warnings

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING']= '1'
warnings.filterwarnings(action='ignore', category=FutureWarning)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--output_dim', type=int, default=16)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--loop', type=bool, default=False)
parser.add_argument('--seed', type=int, default=40)
parser.add_argument('--tf_index', type=int, default=535)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--compare_file', type=str, default='')
args = parser.parse_args()

net_type = "Specific"
cell_type = "mHSC-GM"
num_pair = 268
gene_num = str(500)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATASET_DIR = os.path.join(ROOT_DIR, "Dataset")
exp_file = os.path.join(
    DATASET_DIR, "Benchmark Dataset", f"{net_type} Dataset", cell_type, f"TFs+{gene_num}", "BL--ExpressionData.csv"
)
tf_file = os.path.join(
    DATASET_DIR, "Benchmark Dataset", f"{net_type} Dataset", cell_type, f"TFs+{gene_num}", "TF.csv"
)
target_file = os.path.join(
    DATASET_DIR, "Benchmark Dataset", f"{net_type} Dataset", cell_type, f"TFs+{gene_num}", "Target.csv"
)
train_file = os.path.join(DATASET_DIR, "train", net_type, f"{cell_type} {gene_num}", "Train_set.csv")
val_file = os.path.join(DATASET_DIR, "val", net_type, f"{cell_type} {gene_num}", "Validation_set.csv")
test_file = os.path.join(DATASET_DIR, "test", net_type, f"{cell_type} {gene_num}", "test_set.csv")
# ======================== 读取表达矩阵 ========================
data_input = pd.read_csv(exp_file, index_col=0)
loader = load_data(data_input)
feature = loader.exp_data()
feature = torch.tensor(feature, dtype=torch.float32).to(device)
# ======================== 读取 TF 索引 ========================
tf = pd.read_csv(tf_file, index_col=0)['index'].values.astype(np.int64)
tf = torch.tensor(tf)
# ======================== 构建邻接矩阵 ========================
train_data = pd.read_csv(train_file, index_col=0).values
train_load = scRNADataset(train_data, feature.shape[0])
adj = train_load.Adj_Generate(tf, loop=args.loop)
adj = adj2saprse_tensor(adj).to(device)
# ======================== 验证 & 测试 ========================
val_data = torch.tensor(pd.read_csv(val_file, index_col=0).values).to(device)
test_data = torch.tensor(pd.read_csv(test_file, index_col=0).values).to(device)

#%%
model = MechaGRN(
        input_dim=feature.shape[1],   # cell 数量
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_heads=args.num_heads,
        device=device,
        use_expression=True
    ).to(device)


model_path = os.path.join(ROOT_DIR, "Code", "model", net_type, f"{cell_type} {gene_num}")
state_dict = torch.load(os.path.join(model_path, "best_model.pkl"), map_location=device)
load_msg = model.load_state_dict(state_dict, strict=False)
if args.debug:
    if load_msg.missing_keys:
        print(f"[DEBUG] missing_keys: {load_msg.missing_keys}")
    if load_msg.unexpected_keys:
        print(f"[DEBUG] unexpected_keys: {load_msg.unexpected_keys}")

cs = pd.DataFrame(columns=['TF','Target','Label'])
cs.Target = pd.read_csv(target_file).index
cs.TF = args.tf_index
cs.Label = -1
cs_data = torch.tensor(cs.to_numpy()).to(device)
"""
    HCFC1,329 y
    HDAC2,180 y
    MYC,266 y
    WDR5,232 y
    SPI1,150 y

"""



score_test = model(feature, adj, cs_data)
score_test = torch.nn.functional.sigmoid(score_test)
score_test = score_test.cpu().detach().numpy().round(3)
index = score_test.reshape(-1).argsort()[-num_pair:]
result = cs.iloc[index, :]
result = result.reindex(['TF', 'Target', 'Value'], axis=1)
TF_csv = pd.read_csv(tf_file,index_col=0)
TF_name_index = TF_csv['index'] == result.TF.values[0]
TF_name = TF_csv[TF_name_index].TF.values[0]
Target_csv = pd.read_csv(target_file,index_col=0)
target_name_map = dict(zip(Target_csv['index'].values, Target_csv.iloc[:, 0].values))
result.loc[:, 'TF'] = TF_name
result.loc[:, 'Target'] = result['Target'].map(target_name_map)
result.loc[:, 'Value'] = score_test[index]
result = result.iloc[::-1]
result.index = pd.Index(np.arange(num_pair))
output_file = os.path.join(BASE_DIR, "Regulatory_relationship", f"{cell_type}_{TF_name}.csv")
result.to_csv(output_file)

if args.debug:
    target_indices = cs.iloc[index, :]['Target'].to_numpy()[::-1]
    target_names = result['Target'].to_numpy()
    debug_df = pd.DataFrame({
        'target_index': target_indices,
        'target_name': target_names,
        'score': result['Value'].to_numpy()
    })
    print(f"[DEBUG] TF={TF_name} (index={args.tf_index}) top-{num_pair} predictions")
    print(debug_df.to_string(index=False))
    print(f"[DEBUG] unique target count: {len(set(target_names))}/{num_pair}")

    if args.compare_file:
        compare_df = pd.read_csv(args.compare_file, index_col=0)
        overlap = len(set(target_names) & set(compare_df['Target'].values))
        print(f"[DEBUG] overlap with {args.compare_file}: {overlap}/{num_pair}")
        if overlap > 0:
            overlap_targets = sorted(set(target_names) & set(compare_df['Target'].values))
            print(f"[DEBUG] overlap targets: {overlap_targets}")
    print(f"[DEBUG] output file: {output_file}")


