from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns
# 1. 데이터 불러오기
iris = load_iris()
iris_data = iris['data']

n_clusters_list = [2, 3, 4, 5]

# 전체 그림을 위한 figure와 axes 객체 생성
fig, axs = plt.subplots(len(n_clusters_list), 1, figsize=(8, 4 * len(n_clusters_list)))

# 2. K-Means와 실루엣 분석
for i, n_clusters in enumerate(n_clusters_list):
    # K-Means 모델 초기화
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    
    # 데이터 학습 및 클러스터 라벨 가져오기
    labels = km.fit_predict(iris_data)
    
    # 데이터 포인트별 실루엣 점수 계산
    silhouette_vals = silhouette_samples(iris_data, labels)
    
    # 전체 평균 실루엣 점수 계산
    avg_score = silhouette_score(iris_data, labels)

    # 3. 실루엣 분석 차트 그리기
    y_lower = 10
    
    # 각 클러스터별로 실루엣 점수 시각화
    for j in range(n_clusters):
        # j번째 클러스터에 속한 데이터들의 실루엣 점수 추출 및 정렬
        cluster_silhouette_vals = silhouette_vals[labels == j]
        cluster_silhouette_vals.sort()
        
        # 클러스터 크기
        cluster_size = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + cluster_size
        
        # 색상 설정
        color = cm.nipy_spectral(float(j) / n_clusters)
        
        # 실루엣 막대 그리기
        axs[i].fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,facecolor=color, edgecolor=color, alpha=0.7)
        
        # 클러스터 번호 표시
        axs[i].text(-0.05, y_lower + 0.5 * cluster_size, str(j))
        
        # 다음 클러스터 시작 위치 설정
        y_lower = y_upper + 10

    # 차트 꾸미기
    axs[i].set_title(f"n_clusters = {n_clusters}의 실루엣 분석")
    axs[i].set_xlabel("실루엣 계수")
    axs[i].set_ylabel("클러스터")
    axs[i].set_yticks([])
    axs[i].set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    # 전체 평균 실루엣 점수 나타내는 수직선 그리기
    axs[i].axvline(x=avg_score, color="red", linestyle="--")
    
plt.tight_layout()
plt.show()

# 4. 최적의 군집 수 선택
silhouette_scores = {}
for n_clusters in n_clusters_list:
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = km.fit_predict(iris_data)
    score = silhouette_score(iris_data, labels)
    silhouette_scores[n_clusters] = score

best_n_clusters = max(silhouette_scores, key=silhouette_scores.get)
print(f"\n최적의 군집 수는 실루엣 분석 결과에 따라 {best_n_clusters}일 가능성이 높습니다.")