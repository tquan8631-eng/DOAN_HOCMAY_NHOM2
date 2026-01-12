import matplotlib.pyplot as plt
import pandas as pd

# Class này dùng để đánh giá và so sánh các mô hình phân cụm
class ModelEvaluator:
    def __init__(self, df):
        # Lưu DataFrame để dùng cho các hàm phân tích
        self.df = df

    def compare_distributions(self, cluster_col, feature_col, title):
        """
        Hàm này dùng để so sánh phân phối một thuộc tính (feature)
        giữa các cụm. Mỗi cụm sẽ có một histogram riêng.
        """

        plt.figure(figsize=(10, 6))  # tạo khung hình

        # duyệt qua từng cluster khác -1 (nhiễu)
        for cluster in sorted(self.df[cluster_col].unique()):
            if cluster != -1:  
                subset = self.df[self.df[cluster_col] == cluster]
                plt.hist(
                    subset[feature_col],
                    bins=30,
                    alpha=0.5,
                    label=f"Cluster {cluster}"
                )

        plt.xlabel(feature_col)
        plt.ylabel("So luong khach hang")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def analyze_categorical(self, cluster_col, category_col):
        """
        Hàm phân tích biến phân loại theo từng cụm.
        Ví dụ: cluster → sản phẩm yêu thích / giới tính / nhóm khách hàng.
        """

        print("\n" + "=" * 50)
        print(f"PHAN TICH {category_col.upper()} THEO CUM")
        print("=" * 50)

        ct = pd.crosstab(self.df[cluster_col], self.df[category_col])
        print(ct)

        return ct

    def deep_analysis_by_location(self, cluster_col, category_col):
        """
        Hàm phân tích chi tiết theo Location trong từng cụm.
        Ví dụ: trong cluster 0, khách ở đâu và thích nhóm nào.
        """

        print("\n" + "=" * 60)
        print(f"PHAN TICH {category_col.upper()} THEO LOCATION TRONG TUNG CUM")
        print("=" * 60)

        # Duyệt từng cluster để phân tích riêng
        for cluster in sorted(self.df[cluster_col].unique()):
            if cluster != -1:
                print(f"\n=== CLUSTER {cluster} ===")
                cluster_data = self.df[self.df[cluster_col] == cluster]
                ct = pd.crosstab(cluster_data["Location"], cluster_data[category_col])
                print(ct)

    def summary_table(self, results_dict):
        """
        Tạo bảng tổng hợp kết quả đánh giá (tự truyền vào một dict).
        Dùng để tóm tắt lỗi, điểm silhouette, số cụm...
        """

        print("\n" + "=" * 50)
        print("BANG TONG HOP KET QUA")
        print("=" * 50)

        df_results = pd.DataFrame(results_dict).T
        print(df_results)

        return df_results
