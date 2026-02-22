# TCNM/data/trajectoriesWithMe_unet.py - TEST/INFERENCE VERSION
import os
import logging
import math
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reuse functions from training
from TCNM.data.trajectoriesWithMe_unet_training import (
    env_data_processing, 
    seq_collate
)

class TrajectoryDataset(Dataset):
    """
    Test/Inference Dataset - Đồng bộ với bản Training nhưng hỗ trợ lọc theo năm
    Thứ tự tọa độ: [LONG, LAT, PRES, WIND]
    """
    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002,
                 min_ped=1, delim=' ', other_modal='gph', test_year=None, **kwargs):
        super(TrajectoryDataset, self).__init__()
        
        # Xử lý đường dẫn linh hoạt
        if isinstance(data_dir, dict):
            input_root = data_dir['root']
            dtype = data_dir.get('type', 'test')
        else:
            input_root = data_dir
            dtype = kwargs.get('type', 'test')

        # Đảm bảo trỏ đúng vào thư mục chứa file .txt
        if 'Data1d' in input_root:
            self.data1d_path = input_root if input_root.endswith(dtype) else os.path.join(input_root, dtype)
        else:
            self.data1d_path = os.path.join(input_root, 'Data1d', dtype)
        
        # Thư mục gốc chứa Data3d và ENV_DATA
        self.root_path = os.path.dirname(os.path.dirname(self.data1d_path))
        self.data3d_path = os.path.join(self.root_path, "Data3d")
        self.env_path = os.path.join(self.root_path, "ENV_DATA")
        
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = obs_len + pred_len
        self.modal_name = other_modal

        if not os.path.exists(self.data1d_path):
            logger.warning(f"Data path not found: {self.data1d_path}")
            self.num_seq = 0
            self.tyID = []
            return

        # Lọc file theo năm nếu có yêu cầu
        all_files = [
            os.path.join(self.data1d_path, f) 
            for f in os.listdir(self.data1d_path) 
            if (not test_year or str(test_year) in f) and f.endswith('.txt')
        ]
        
        logger.info(f"Found {len(all_files)} test files for year {test_year}")
        
        self.seq_list = []
        self.seq_list_rel = []
        self.non_linear_ped = []
        self.tyID = []
        num_peds_in_seq = []

        for path in all_files:
            filename = os.path.basename(path).replace('.txt', '')
            parts = filename.split('_')
            
            if len(parts) >= 2:
                f_year, f_name = parts[0], parts[1]
            else:
                f_year, f_name = "Unknown", filename

            data_dict = self._read_track_file(path)
            addinf, data = data_dict['addition'], data_dict['main']
            
            if len(data) < self.seq_len:
                continue
            
            frames = np.unique(data[:, 0]).tolist()
            frame_data = [data[data[:, 0] == f, :] for f in frames]
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / self.skip))

            for idx in range(0, num_sequences * self.skip, self.skip):
                if idx + self.seq_len > len(frame_data):
                    break
                
                curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                peds = np.unique(curr_seq_data[:, 1])
                count = 0
                
                c_seq = np.zeros((len(peds), 4, self.seq_len))
                c_rel = np.zeros((len(peds), 4, self.seq_len))
                
                for p_id in peds:
                    p_seq = curr_seq_data[curr_seq_data[:, 1] == p_id, :]
                    if len(p_seq) != self.seq_len:
                        continue
                    
                    # p_seq[:, 2:] chứa [LONG, LAT, PRES, WIND]
                    p_seq = np.transpose(p_seq[:, 2:])
                    
                    rel_seq = np.zeros(p_seq.shape)
                    rel_seq[:, 1:] = p_seq[:, 1:] - p_seq[:, :-1]
                    
                    c_seq[count] = p_seq
                    c_rel[count] = rel_seq
                    self.non_linear_ped.append(1.0)
                    count += 1

                if count >= min_ped:
                    num_peds_in_seq.append(count)
                    self.seq_list.append(c_seq[:count])
                    self.seq_list_rel.append(c_rel[:count])
                    self.tyID.append({
                        'old': [f_year, f_name, idx], 
                        'tydate': [addinf[i][0] for i in range(idx, idx + self.seq_len)]
                    })

        self.num_seq = len(self.seq_list)
        
        if self.num_seq > 0:
            all_seq = np.concatenate(self.seq_list, axis=0)
            all_rel = np.concatenate(self.seq_list_rel, axis=0)
            
            # Cắt dữ liệu theo quan sát và dự báo
            # Trục 1: [0:Long, 1:Lat] | [2:Pres, 3:Wind]
            self.obs_traj = torch.from_numpy(all_seq[:, :2, :self.obs_len]).float()
            self.pred_traj = torch.from_numpy(all_seq[:, :2, self.obs_len:]).float()
            self.obs_Me = torch.from_numpy(all_seq[:, 2:, :self.obs_len]).float()
            self.pred_Me = torch.from_numpy(all_seq[:, 2:, self.obs_len:]).float()
            
            self.obs_rel = torch.from_numpy(all_rel[:, :2, :self.obs_len]).float()
            self.pred_rel = torch.from_numpy(all_rel[:, :2, self.obs_len:]).float()
            
            cumsum = np.cumsum(num_peds_in_seq).tolist()
            self.seq_start_end = [(s, e) for s, e in zip([0] + cumsum[:-1], cumsum)]
            
            logger.info(f"✅ Loaded {self.num_seq} sequences for inference")

    def _read_track_file(self, path):
        """Đọc file .txt với định dạng: STT, 1.0, LONG, LAT, PRES, WIND, DATE, NAME"""
        data, add = [], []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 6: continue
                
                # Cấu trúc: [Frame_ID, Ped_ID, Long, Lat, Pres, Wind]
                row = [float(i), 1.0]
                for val in parts[2:6]:
                    try:
                        row.append(float(val) if val.lower() != 'null' else 0.0)
                    except:
                        row.append(0.0)
                data.append(row)
                add.append(parts[-2:]) # DATE và NAME
        return {'main': np.asarray(data), 'addition': add}

    def _load_data3d(self, year, ty_name, timestamp):
        """Load ảnh vệ tinh từ Data3d/YEAR/TY_NAME/"""
        # Thử tìm file với prefix WP hoặc không có prefix
        possible_names = [f"WP{year}{ty_name}_{timestamp}.npy", f"{timestamp}.npy"]
        img = None
        
        for name in possible_names:
            path = os.path.join(self.data3d_path, str(year), ty_name, name)
            if os.path.exists(path):
                img = np.load(path)
                break
        
        if img is None:
            return torch.zeros((64, 64, 1))
        
        img = cv2.resize(img, (64, 64))
        img = self._transforms(img)
        return torch.from_numpy(img[:, :, np.newaxis]).float()

    def _load_env_data(self, year, ty_name, timestamp):
        """Load dict môi trường từ ENV_DATA/YEAR/TY_NAME/"""
        possible_names = [f"WP{year}{ty_name}_{timestamp}.npy", f"{timestamp}.npy"]
        for name in possible_names:
            path = os.path.join(self.env_path, str(year), ty_name, name)
            if os.path.exists(path):
                try:
                    data = np.load(path, allow_pickle=True).item()
                    # Clean data: replace -1 with 0
                    for k in data:
                        if isinstance(data[k], (int, float)) and data[k] == -1:
                            data[k] = 0.0
                    return data
                except: continue
        
        return {'wind': 0.0, 'move_velocity': 0.0, 'month': np.zeros(12)}

    def _transforms(self, img):
        modal_range = {'gph': (44490.5, 58768.4), 'sst': (273, 312)}
        all_min, all_max = modal_range.get(self.modal_name, (img.min(), img.max() + 1e-6))
        img = (img - all_min) / (all_max - all_min)
        return np.clip(img, 0, 1)

    def _embed_time(self, date_list):
        data_embed = []
        for date in date_list:
            y = (float(date[:4]) - 1949) / 76.0 - 0.5
            m = (float(date[4:6]) - 1) / 11.0 - 0.5
            d = (float(date[6:8]) - 1) / 30.0 - 0.5
            h = float(date[8:10]) / 18.0 - 0.5
            data_embed.append([y, m, d, h])
        return torch.tensor(data_embed).transpose(1, 0).unsqueeze(0).float()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        s, e = self.seq_start_end[index]
        info = self.tyID[index]
        dates = info['tydate']
        year, ty_name = info['old'][0], info['old'][1]
        
        # Tọa độ và cường độ [T, 2]
        obs_traj = self.obs_traj[s:e].squeeze(0)
        pred_traj = self.pred_traj[s:e].squeeze(0)
        obs_Me = self.obs_Me[s:e].squeeze(0)
        pred_Me = self.pred_Me[s:e].squeeze(0)
        
        # Relative data (cần cho một số thành phần model cũ)
        obs_rel = self.obs_rel[s:e].squeeze(0)
        pred_rel = self.pred_rel[s:e].squeeze(0)
        
        # Multimodal data
        img_obs_list = [self._load_data3d(year, ty_name, d) for d in dates[:self.obs_len]]
        img_obs = torch.stack(img_obs_list, dim=0) # [T_obs, 64, 64, 1]
        
        # Chỉ load env_data tại thời điểm hiện tại (cuối obs_len)
        env_dict = self._load_env_data(year, ty_name, dates[self.obs_len-1])
        
        # Cấu trúc trả về 16 phần tử khớp với seq_collate
        return [
            obs_traj, pred_traj, obs_rel, pred_rel,
            1.0, torch.ones((self.seq_len)), # non_lin, mask
            obs_Me, pred_Me, obs_rel, pred_rel,
            self._embed_time(dates[:self.obs_len]), 
            self._embed_time(dates[self.obs_len:]),
            img_obs, 
            torch.zeros((self.pred_len, 64, 64, 1)), # img_pre (dummy)
            env_dict, 
            info
        ]