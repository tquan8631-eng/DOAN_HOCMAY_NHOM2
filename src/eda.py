import matplotlib.pyplot as plt
import pandas as pd

# Class này dùng để gom các hàm phân tích dữ liệu lại cho gọn.
class EDA:
    def __init__(self, df):
        # Lưu DataFrame lại để dùng cho các hàm khác
        self.df = df

    def analyze_distribution(self, column, title=None):
        """
        Hàm xem phân phối dữ liệu của một cột dạng số
        bằng histogram để biết nó trải rộng hay lệch như thế nào.
        """
        plt.figure(figsize=(8, 5))  # tạo khung hình

        plt.hist(
            self.df[column],
            bins=30,           # chia nhỏ cho mượt
            alpha=0.7,         # độ trong suốt nhẹ
            edgecolor="black"  # viền đen cho rõ
        )

        plt.xlabel(column)        # nhãn trục X
        plt.ylabel("So luong")    # nhãn trục Y
        plt.title(title or f"Phan phoi cua {column}")  # tiêu đề
        plt.grid(True, alpha=0.3)  # bật grid
        plt.show()

    def analyze_category(self, column):
        """
        Hàm phân tích cột phân loại (category), đếm số lượng
        từng nhóm trong cột, ví dụ: Male/Female, Yes/No.
        """
        print("\n" + "=" * 50)
        print(f"PHAN TICH COT: {column.upper()}")
        print("=" * 50)

        counts = self.df[column].value_counts()  # đếm từng nhóm
        print(counts)
        return counts

    def analyze_crosstab(self, col1, col2):
        """
        Hàm tạo bảng chéo giữa hai biến phân loại để xem mối liên hệ
        giữa chúng như thế nào.
        """
        print(f"\nCROSSTAB: {col1} vs {col2}")

        ct = pd.crosstab(self.df[col1], self.df[col2])  # tạo bảng chéo
        print(ct)
        return ct
