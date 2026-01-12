
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import numpy as np

class KMeansClusterer:
    """Phân cụm bằng K-Means"""
    
    def __init__(self):
        self.model = None
        self.labels = None
        self.n_clusters = None
        
    def elbow_method(self, X_scaled, k_range=range(1, 11)):
        """Tìm số cụm tối ưu bằng Elbow Method"""
        print("\n" + "="*50)
        print("ELBOW METHOD - XAC DINH SO CUM")
        print("="*50)
        
        wcss = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)
            print(f"k={k}: WCSS={kmeans.inertia_:.2f}")
        
        plt.figure(figsize=(8, 5))
        plt.plot(k_range, wcss, marker='o', linewidth=2)
        plt.xlabel("So cum (k)")
        plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
        plt.title("Elbow Method - Xac dinh so cum toi uu")
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return wcss
    
    def silhouette_analysis(self, X_scaled, k_range=range(2, 7)):
        """Phân tích Silhouette Score"""
        print("\n" + "="*50)
        print("SILHOUETTE ANALYSIS")
        print("="*50)
        
        scores = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            scores.append(score)
            print(f"k={k}: Silhouette Score={score:.4f}")
        
        plt.figure(figsize=(8, 5))
        plt.plot(k_range, scores, marker='o', linewidth=2, color='green')
        plt.xlabel("So cum (k)")
        plt.ylabel("Silhouette Score")
        plt.title("So sanh Silhouette Score theo so cum")
        plt.grid(True, alpha=0.3)
        plt.show()
        
        best_k = list(k_range)[np.argmax(scores)]
        print(f"\nSo cum toi uu (Silhouette): k={best_k}")
        
        return scores
    
    def fit(self, X_scaled, n_clusters=3):
        """Huấn luyện mô hình K-Means"""
        print(f"\n" + "="*50)
        print(f"HUAN LUYEN K-MEANS VOI {n_clusters} CUM")
        print("="*50)
        
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.labels = self.model.fit_predict(X_scaled)
        
        print(f"Hoan thanh huan luyen")
        print(f"So diem moi cum:")
        unique, counts = np.unique(self.labels, return_counts=True)
        for cluster, count in zip(unique, counts):
            print(f"   - Cluster {cluster}: {count} diem")
        
        return self.labels
    
    def evaluate(self, X_scaled):
        """Đánh giá mô hình"""
        print(f"\n" + "="*50)
        print("DANH GIA MO HINH K-MEANS")
        print("="*50)
        
        sil_score = silhouette_score(X_scaled, self.labels)
        db_score = davies_bouldin_score(X_scaled, self.labels)
        
        print(f"\nCac chi so danh gia:")
        print(f"   - Silhouette Score: {sil_score:.4f} (cao hon tot hon, max=1)")
        print(f"   - Davies-Bouldin Index: {db_score:.4f} (thap hon tot hon)")
        
        return {'silhouette': sil_score, 'davies_bouldin': db_score}
    
    def visualize(self, df, title="Phan cum K-Means"):
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
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel("Purchase Amount (USD)")
        plt.ylabel("Frequency (Numeric)")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def analyze_clusters(self, df):
        """Phân tích đặc trưng các cụm"""
        print(f"\n" + "="*50)
        print("PHAN TICH DOC DIEM CAC CUM")
        print("="*50)
        
        df_temp = df.copy()
        df_temp['Cluster'] = self.labels
        
        summary = df_temp.groupby('Cluster')[['Purchase Amount (USD)', 'Frequency_Num']].agg(['mean', 'std', 'min', 'max'])
        print(summary)
        
        return summary