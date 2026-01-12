
import sys

import pandas as pd
from preprocessing import DataPreprocessor
from eda import EDA
from feature_engineering import FeatureEngineer
from model_kmeans_lethilinh import KMeansClusterer
from model_dbscan_tranminhquan import DBSCANClusterer
from model_hierarchical_votruongan import HierarchicalClusterer
from evaluation import ModelEvaluator

def main():
    """Hàm chính chạy toàn bộ quy trình"""
    
    print("="*60)
    print("   DU AN PHAN CUM KHACH HANG THEO HANH VI MUA SAM")
    print("="*60)
    
    # BUOC 1: LOAD VA XU LY DU LIEU
    filepath = input(r"\nNhap duong dan file CSV (hoac Enter de dung mac dinh): ").strip()
    if not filepath:
        # Duong dan mac dinh cho Google Colab
        filepath = r"C:\Users\Quan\Downloads\shopping_behavior_updated.csv"
    
    print(f"\nDang xu ly file: {filepath}")
    
    preprocessor = DataPreprocessor(filepath) # Khởi tạo bộ tiền xử lý
    df = preprocessor.load_data()  # Load dữ liệu CSV vào DataFrame
    preprocessor.check_quality()  # Kiểm tra chất lượng dữ liệu
    df = preprocessor.map_frequency() # Map Frequency of Purchases -> Frequency_Num
    X = preprocessor.get_features()  # Lấy tập đặc trưng đầu vào (X)
    
    # BUOC 2: PHAN TICH KHAM PHA (EDA)
    print("\n" + "="*60)
    print("BUOC 2: PHAN TICH KHAM PHA DU LIEU (EDA)")
    print("="*60)
    
    eda = EDA(df) # Khởi tạo đối tượng EDA
    eda.analyze_distribution('Purchase Amount (USD)', 'Phan phoi so tien mua hang')  # Phân phối mức chi tiêu
    eda.analyze_distribution('Frequency_Num', 'Phan phoi tan suat mua')  # Phân phối tần suất mua hàng
    
    # BUOC 3: CHUAN HOA DU LIEU
    print("\n" + "="*60)
    print("BUOC 3: CHUAN HOA DAC TRUNG")
    print("="*60)
    
    fe = FeatureEngineer()  # Khởi tạo module xử lý đặc trưng
    X_scaled = fe.scale_features(X) # Chuẩn hóa dữ liệu
    
    # BUOC 4: PHAN CUM K-MEANS
    print("\n" + "="*60)
    print("BUOC 4: PHAN CUM K-MEANS")
    print("="*60)
    
    kmeans = KMeansClusterer()  # Khởi tạo mô hình K-Means
    kmeans.elbow_method(X_scaled)  # Áp dụng Elbow Method xác định số cụm
    kmeans.silhouette_analysis(X_scaled) # Phân tích Silhouette theo số cụm
    
    labels_kmeans = kmeans.fit(X_scaled, n_clusters=3)  # Huấn luyện K-Means với k = 3
    df['Cluster'] = labels_kmeans   # Lưu nhãn cụm K-Means vào DataFrame
    
    results_kmeans = kmeans.evaluate(X_scaled)  # Đánh giá chất lượng K-Means
    kmeans.visualize(df, "Phan cum khach hang - K-Means") # Trực quan hóa kết quả phân cụm K-Means
    kmeans.analyze_clusters(df)  # Phân tích đặc trưng từng cụm K-Means
    
    # BUOC 5: PHAN CUM DBSCAN
    print("\n" + "="*60)
    print("BUOC 5: PHAN CUM DBSCAN")
    print("="*60)
    
    dbscan = DBSCANClusterer(eps=0.7, min_samples=5)  # Tham số mật độ DBSCAN
    labels_dbscan = dbscan.fit(X_scaled)  # Huấn luyện DBSCAN
    df['Cluster_DBSCAN'] = labels_dbscan # Lưu nhãn DBSCAN (-1 là nhiễu)
    
    results_dbscan = dbscan.evaluate(X_scaled) # Đánh giá DBSCAN
    dbscan.visualize(df, "Phan cum khach hang - DBSCAN")  # Trực quan hóa kết quả DBSCAN
    
    # BUOC 6: PHAN CUM HIERARCHICAL
    print("\n" + "="*60)
    print("BUOC 6: PHAN CUM PHAN CAP (HIERARCHICAL)")
    print("="*60)
    
    hc = HierarchicalClusterer(n_clusters=3, linkage='ward')  # Khởi tạo phân cụm phân cấp
    hc.plot_dendrogram(X_scaled)  # Vẽ dendrogram quan sát cấu trúc dữ liệu
    labels_hc = hc.fit(X_scaled) # Huấn luyện Hierarchical Clustering
    df['Cluster_HC'] = labels_hc   # Lưu nhãn cụm phân cấp
    
    results_hc = hc.evaluate(X_scaled)  # Đánh giá phân cụm phân cấp
    hc.visualize(df, "Phan cum khach hang - Hierarchical")  # Trực quan hóa kết quả Hierarchical
    hc.analyze_clusters(df)  # Phân tích đặc trưng từng cụm
    
    # BUOC 7: SO SANH VA DANH GIA
    print("\n" + "="*60)
    print("BUOC 7: SO SANH VA DANH GIA CAC MO HINH")
    print("="*60)
    
    evaluator = ModelEvaluator(df)  # Khởi tạo bộ đánh giá tổng hợp
    
    # So sanh phan phoi chi tieu
    evaluator.compare_distributions('Cluster', 'Purchase Amount (USD)', 
                                   'Phan phoi chi tieu theo cum - K-Means')
    
    # So sanh phan phoi tan suat mua
    evaluator.compare_distributions('Cluster', 'Frequency_Num', 
                                   'Phan phoi tan suat mua theo cum - K-Means')
    
    # Phan tich cac bien phan loai
    evaluator.analyze_categorical('Cluster', 'Size')
    evaluator.analyze_categorical('Cluster', 'Color')
    evaluator.analyze_categorical('Cluster', 'Season')
    evaluator.analyze_categorical('Cluster', 'Location')
    
    # Phan tich chi tiet theo Location
    evaluator.deep_analysis_by_location('Cluster', 'Size')
    evaluator.deep_analysis_by_location('Cluster', 'Color')
    evaluator.deep_analysis_by_location('Cluster', 'Season')
    
    # Bang tong hop ket qua
    results_dict = {
        'K-Means': results_kmeans,
        'DBSCAN': results_dbscan,
        'Hierarchical': results_hc
    }
    evaluator.summary_table(results_dict)
    
    print("\n" + "="*60)
    print("HOAN THANH PHAN TICH!")
    print("="*60)
    
    # Luu ket qua
    save_option = input("\nBan co muon luu ket qua khong? (y/n): ").strip().lower()
    if save_option == 'y':
        output_path = input("Nhap duong dan luu file (vd: /content/drive/MyDrive/results.csv): ").strip()
        df.to_csv(output_path, index=False)
        print(f"Da luu ket qua vao: {output_path}")

if __name__ == "__main__":
    main()