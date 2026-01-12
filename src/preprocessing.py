import pandas as pd
import numpy as np

class DataPreprocessor:
    """Xử lý và làm sạch dữ liệu"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        
    def load_data(self):
        """Đọc dữ liệu từ CSV"""
        try:
            self.df = pd.read_csv(self.filepath)
            print(f"Da load {len(self.df)} dong du lieu")
            return self.df
        except Exception as e:
            print(f"Loi khi doc file: {e}")
            raise
    
    def check_quality(self):
        """Kiểm tra chất lượng dữ liệu"""
        print("\n" + "="*50)
        print("KIEM TRA CHAT LUONG DU LIEU")
        print("="*50)
        
        print(f"\nThong tin co ban:")
        print(f"   - So dong: {len(self.df)}")
        print(f"   - So cot: {len(self.df.columns)}")
        
        print(f"\nThong tin chi tiet:")
        self.df.info()
        
        print(f"\nThong ke mo ta:")
        print(self.df.describe())
        
        null_counts = self.df.isnull().sum()
        if null_counts.sum() > 0:
            print(f"\nGia tri null:")
            print(null_counts[null_counts > 0])
        else:
            print(f"\nKhong co gia tri null")
        
        dup_count = self.df.duplicated().sum()
        print(f"\nSo dong trung lap: {dup_count}")
        
    def map_frequency(self):
        """Chuyển đổi tần suất mua từ text sang số"""
        print("\n" + "="*50)
        print("CHUYEN DOI TAN SUAT MUA HANG")
        print("="*50)
        
        # Kiểm tra giá trị unique
        print("\nCac gia tri tan suat hien co:")
        print(self.df['Frequency of Purchases'].unique())
        
        # Ánh xạ sang số (lần mua/tháng)
        freq_map = {
            'Daily': 30,
            'Weekly': 4,
            'Bi-Weekly': 2,
            'Fortnightly': 2,
            'Monthly': 1,
            'Every 3 Months': 0.33,
            'Quarterly': 0.25,
            'Annually': 0.08
        }
        
        self.df['Frequency_Num'] = self.df['Frequency of Purchases'].map(freq_map)
        
        # Kiểm tra NaN
        nan_count = self.df['Frequency_Num'].isna().sum()
        if nan_count > 0:
            print(f"\nPhat hien {nan_count} gia tri khong anh xa duoc")
            self.df = self.df.dropna(subset=['Frequency_Num'])
            print(f"Da xoa cac dong NaN, con lai {len(self.df)} dong")
        else:
            print(f"\nAnh xa thanh cong, khong co NaN")
            
        # Hiển thị kết quả
        print(f"\nKet qua anh xa (5 dong dau):")
        print(self.df[['Frequency of Purchases', 'Frequency_Num']].head())
        
        return self.df
    
    def get_features(self):
        """Lấy các đặc trưng để phân cụm"""
        X = self.df[['Purchase Amount (USD)', 'Frequency_Num']]
        print(f"\nDa trich xuat {X.shape[1]} dac trung")
        return X
    
    def get_dataframe(self):
        """Trả về dataframe đã xử lý"""
        return self.df