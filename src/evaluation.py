
import matplotlib.pyplot as plt
import pandas as pd

class ModelEvaluator:
    """Đánh giá và so sánh các mô hình phân cụm"""
    
    def __init__(self, df):
        self.df = df
        
    def compare_distributions(self, cluster_col, feature_col, title):
        """So sánh phân phối theo cụm"""
        plt.figure(figsize=(10, 6))
        
        for cluster in sorted(self.df[cluster_col].unique()):
            if cluster != -1:  # Bỏ qua nhiễu nếu có
                subset = self.df[self.df[cluster_col] == cluster]
                plt.hist(subset[feature_col], alpha=0.5, label=f'Cluster {cluster}', bins=30)
        
        plt.xlabel(feature_col)
        plt.ylabel("So luong khach hang")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def analyze_categorical(self, cluster_col, category_col):
        """Phân tích biến phân loại theo cụm"""
        print(f"\n{'='*50}")
        print(f"PHAN TICH {category_col.upper()} THEO CUM")
        print('='*50)
        
        ct = pd.crosstab(self.df[cluster_col], self.df[category_col])
        print(ct)
        return ct
    
    def deep_analysis_by_location(self, cluster_col, category_col):
        """Phân tích chi tiết theo Location"""
        print(f"\n{'='*60}")
        print(f"PHAN TICH {category_col.upper()} THEO LOCATION TRONG TUNG CUM")
        print('='*60)
        
        for cluster in sorted(self.df[cluster_col].unique()):
            if cluster != -1:
                print(f"\n=== CLUSTER {cluster} ===")
                cluster_data = self.df[self.df[cluster_col] == cluster]
                ct = pd.crosstab(cluster_data['Location'], cluster_data[category_col])
                print(ct)
    
    def summary_table(self, results_dict):
        """Tạo bảng tổng hợp kết quả"""
        print(f"\n{'='*50}")
        print("BANG TONG HOP KET QUA")
        print('='*50)
        
        df_results = pd.DataFrame(results_dict).T
        print(df_results)
        
        return df_results