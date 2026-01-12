from sklearn.preprocessing import StandardScaler
import numpy as np

class FeatureEngineer:
    """Xử lý đặc trưng"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.X_scaled = None
        
    def scale_features(self, X):
        """Chuẩn hóa dữ liệu bằng StandardScaler"""
        print("\n" + "="*50)
        print("CHUAN HOA DU LIEU")
        print("="*50)
        
        print(f"\nTruoc khi chuan hoa:")
        print(f"   - Mean: {X.mean().values}")
        print(f"   - Std: {X.std().values}")
        
        self.X_scaled = self.scaler.fit_transform(X)
        
        print(f"\nSau khi chuan hoa:")
        print(f"   - Mean: ~{np.mean(self.X_scaled, axis=0)}")
        print(f"   - Std: ~{np.std(self.X_scaled, axis=0)}")
        
        return self.X_scaled
    
    def get_scaled_data(self):
        """Trả về dữ liệu đã chuẩn hóa"""
        return self.X_scaled