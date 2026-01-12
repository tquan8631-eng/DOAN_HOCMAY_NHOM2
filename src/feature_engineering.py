from sklearn.preprocessing import StandardScaler
import numpy as np

# Class này mình dùng để xử lý đặc trưng, chủ yếu là chuẩn hóa dữ liệu
class FeatureEngineer:
    def __init__(self):
        # Tạo scaler để tái sử dụng khi cần
        self.scaler = StandardScaler()
        self.X_scaled = None

    def scale_features(self, X):
        """
        Hàm chuẩn hóa dữ liệu về dạng mean = 0, std = 1.
        Giúp mô hình học tốt hơn, nhất là mấy thuật toán phân cụm.
        """

        print("\n" + "="*50)
        print("CHUAN HOA DU LIEU")
        print("="*50)

        # In thông tin trước khi chuẩn hóa
        print("\nTruoc khi chuan hoa:")
        print(f"   - Mean: {X.mean().values}")
        print(f"   - Std : {X.std().values}")

        # Thực hiện chuẩn hóa
        self.X_scaled = self.scaler.fit_transform(X)

        # In thông tin sau khi chuẩn hóa
        print("\nSau khi chuan hoa:")
        print(f"   - Mean: ~{np.mean(self.X_scaled, axis=0)}")
        print(f"   - Std : ~{np.std(self.X_scaled, axis=0)}")

        return self.X_scaled

    def get_scaled_data(self):
        """
        Trả về dữ liệu đã được chuẩn hóa.
        Chủ yếu để tái sử dụng trong pipeline.
        """
        return self.X_scaled
