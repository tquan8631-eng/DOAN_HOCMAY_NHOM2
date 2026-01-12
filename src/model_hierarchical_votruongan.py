
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np

class HierarchicalClusterer:
    """Phân cụm phân cấp"""
    
    def __init__(self, n_clusters=3, linkage='ward'):
        self.n_clusters = n_clusters
        self.linkage_method = linkage
        self.model = None
        self.labels = None
        
    def plot_dendrogram(self, X_scaled):
        """Vẽ dendrogram"""
        print(f"\n" + "="*50)
        print("VE DENDROGRAM")
        print("="*50)
        
        linked = linkage(X_scaled, method=self.linkage_method)
        
        plt.figure(figsize=(12, 6))
        dendrogram(linked)
        plt.xlabel("Chi so khach hang")
        plt.ylabel("Khoang cach")
        plt.title(f"Dendrogram - Hierarchical Clustering ({self.linkage_method})")
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print("Da ve dendrogram")
        
    def fit(self, X_scaled):
        """Huấn luyện mô hình"""
        print(f"\n" + "="*50)
        print(f"HUAN LUYEN HIERARCHICAL CLUSTERING")
        print("="*50)
        print(f"   - So cum: {self.n_clusters}")
        print(f"   - Linkage: {self.linkage_method}")
        
        self.model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=self.linkage_method
        )
        self.labels = self.model.fit_predict(X_scaled)
        
        unique, counts = np.unique(self.labels, return_counts=True)
        print(f"\nHoan thanh phan cum:")
        for cluster, count in zip(unique, counts):
            print(f"   - Cluster {cluster}: {count} diem")
        
        return self.labels
    
    def evaluate(self, X_scaled):
        """Đánh giá mô hình"""
        print(f"\n" + "="*50)
        print("DANH GIA MO HINH HIERARCHICAL")
        print("="*50)
        
        sil_score = silhouette_score(X_scaled, self.labels)
        print(f"   - Silhouette Score: {sil_score:.4f}")
        
        return {'silhouette': sil_score}
    
    def visualize(self, df, title="Phan cum Hierarchical"):
        """Trực quan hóa kết quả"""
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            df['Purchase Amount (USD)'],
            df['Frequency_Num'],
            c=self.labels,
            cmap='tab10',
            s=50,
            alpha=0.6,
            edgecolors='black'
        )
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel("Purchase Amount (USD)")
        plt.ylabel("Frequency (Numeric)")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def analyze_clusters(self, df):
        """Phân tích các cụm"""
        df_temp = df.copy()
        df_temp['Cluster_HC'] = self.labels
        
        summary = df_temp.groupby('Cluster_HC')[['Purchase Amount (USD)', 'Frequency_Num']].mean()
        print(f"\nDac trung trung binh cac cum:")
        print(summary)
        
        return summary