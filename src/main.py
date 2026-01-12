
import sys
import pandas as pd
from preprocessing import DataPreprocessor
from eda import EDA
from feature_engineering import FeatureEngineer
from model_kmeans import KMeansClusterer
from model_dbscan import DBSCANClusterer
from model_hierarchical import HierarchicalClusterer
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
    
    preprocessor = DataPreprocessor(filepath)
    df = preprocessor.load_data()
    preprocessor.check_quality()
    df = preprocessor.map_frequency()
    X = preprocessor.get_features()
    
    # BUOC 2: PHAN TICH KHAM PHA (EDA)
    print("\n" + "="*60)
    print("BUOC 2: PHAN TICH KHAM PHA DU LIEU (EDA)")
    print("="*60)
    
    eda = EDA(df)
    eda.analyze_distribution('Purchase Amount (USD)', 'Phan phoi so tien mua hang')
    eda.analyze_distribution('Frequency_Num', 'Phan phoi tan suat mua')
    
    # BUOC 3: CHUAN HOA DU LIEU
    print("\n" + "="*60)
    print("BUOC 3: CHUAN HOA DAC TRUNG")
    print("="*60)
    
    fe = FeatureEngineer()
    X_scaled = fe.scale_features(X)
    
    # BUOC 4: PHAN CUM K-MEANS
    print("\n" + "="*60)
    print("BUOC 4: PHAN CUM K-MEANS")
    print("="*60)
    
    kmeans = KMeansClusterer()
    kmeans.elbow_method(X_scaled)
    kmeans.silhouette_analysis(X_scaled)
    
    labels_kmeans = kmeans.fit(X_scaled, n_clusters=3)
    df['Cluster'] = labels_kmeans
    
    results_kmeans = kmeans.evaluate(X_scaled)
    kmeans.visualize(df, "Phan cum khach hang - K-Means")
    kmeans.analyze_clusters(df)
    
    # BUOC 5: PHAN CUM DBSCAN
    print("\n" + "="*60)
    print("BUOC 5: PHAN CUM DBSCAN")
    print("="*60)
    
    dbscan = DBSCANClusterer(eps=0.7, min_samples=5)
    labels_dbscan = dbscan.fit(X_scaled)
    df['Cluster_DBSCAN'] = labels_dbscan
    
    results_dbscan = dbscan.evaluate(X_scaled)
    dbscan.visualize(df, "Phan cum khach hang - DBSCAN")
    
    # BUOC 6: PHAN CUM HIERARCHICAL
    print("\n" + "="*60)
    print("BUOC 6: PHAN CUM PHAN CAP (HIERARCHICAL)")
    print("="*60)
    
    hc = HierarchicalClusterer(n_clusters=3, linkage='ward')
    hc.plot_dendrogram(X_scaled)
    labels_hc = hc.fit(X_scaled)
    df['Cluster_HC'] = labels_hc
    
    results_hc = hc.evaluate(X_scaled)
    hc.visualize(df, "Phan cum khach hang - Hierarchical")
    hc.analyze_clusters(df)
    
    # BUOC 7: SO SANH VA DANH GIA
    print("\n" + "="*60)
    print("BUOC 7: SO SANH VA DANH GIA CAC MO HINH")
    print("="*60)
    
    evaluator = ModelEvaluator(df)
    
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