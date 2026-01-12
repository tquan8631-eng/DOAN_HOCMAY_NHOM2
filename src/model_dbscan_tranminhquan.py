
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

class DBSCANClusterer:
    """Phân cụm bằng DBSCAN"""
    
    def __init__(self, eps=0.7, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.model = None
        self.labels = None
        
    def fit(self, X_scaled):
        """Huấn luyện mô hình DBSCAN"""
        print(f"\n" + "="*50)
        print(f"HUAN LUYEN DBSCAN")
        print("="*50)
        print(f"   - eps (ban kinh): {self.eps}")
        print(f"   - min_samples: {self.min_samples}")
        
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels = self.model.fit_predict(X_scaled)
        
        # Thống kê
        unique, counts = np.unique(self.labels, return_counts=True)
        n_clusters = len(unique[unique != -1])
        n_noise = counts[unique == -1][0] if -1 in unique else 0
        
        print(f"\nHoan thanh phan cum:")
        print(f"   - So cum tim duoc: {n_clusters}")
        print(f"   - So diem nhieu (outliers): {n_noise}")
        print(f"\nPhan bo:")
        for cluster, count in zip(unique, counts):
            if cluster == -1:
                print(f"   - Nhieu: {count} diem")
            else:
                print(f"   - Cluster {cluster}: {count} diem")
        
        return self.labels
    
    def evaluate(self, X_scaled):
        """Đánh giá mô hình DBSCAN"""
        print(f"\n" + "="*50)
        print("DANH GIA MO HINH DBSCAN")
        print("="*50)
        
        unique_labels = set(self.labels)
        
        if len(unique_labels) > 1 and -1 not in unique_labels:
            sil_score = silhouette_score(X_scaled, self.labels)
            print(f"   - Silhouette Score: {sil_score:.4f}")
            return {'silhouette': sil_score}
        else:
            print("   Khong the tinh Silhouette vi co nhieu hoac chi co 1 cum")
            return {'silhouette': None}
    
    def visualize(self, df, title="Phan cum DBSCAN"):
        """Trực quan hóa kết quả"""
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            df['Purchase Amount (USD)'],
            df['Frequency_Num'],
            c=self.labels,
            cmap='viridis',
            s=50,
            alpha=0.6,
            edgecolors='black'
        )
        plt.colorbar(scatter, label='Cluster (-1 = Nhieu)')
        plt.xlabel("Purchase Amount (USD)")
        plt.ylabel("Frequency (Numeric)")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()