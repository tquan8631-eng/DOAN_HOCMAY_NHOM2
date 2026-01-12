import matplotlib.pyplot as plt
import pandas as pd

class EDA:
    """Phân tích khám phá dữ liệu"""
    
    def __init__(self, df):
        self.df = df
        
    def analyze_distribution(self, column, title=None):
        """Phân tích phân phối của một cột"""
        plt.figure(figsize=(8, 5))
        plt.hist(self.df[column], bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel(column)
        plt.ylabel('So luong')
        plt.title(title or f'Phan phoi {column}')
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def analyze_category(self, column):
        """Phân tích biến phân loại"""
        print(f"\n{'='*50}")
        print(f"PHAN TICH {column.upper()}")
        print('='*50)
        counts = self.df[column].value_counts()
        print(counts)
        return counts
    
    def analyze_crosstab(self, col1, col2):
        """Phân tích chéo giữa 2 biến"""
        print(f"\nCrosstab: {col1} vs {col2}")
        ct = pd.crosstab(self.df[col1], self.df[col2])
        print(ct)
        return ct