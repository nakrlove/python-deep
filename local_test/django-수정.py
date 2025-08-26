import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import FileUtils  
# # 가정: FileUtils() 클래스는 CSV 파일을 불러오고 이미지를 저장하는 기능을 제공합니다.
# # 실제 Django 환경에서는 FileUtils 클래스를 구현하거나, 이미지 저장 경로를 설정해야 합니다.
# class FileUtils:
#     def getCsv(self, file_path):
#         try:
#             return pd.read_csv(file_path, encoding='utf-8', sep=';')
#         except UnicodeDecodeError:
#             return pd.read_csv(file_path, encoding='euc-kr', sep=';')
    
#     def savePngToPath(self, filename, closeFlag=True):
#         save_path = f'static/images/{filename}'
#         plt.savefig(save_path)
#         if closeFlag:
#             plt.close()
#         return save_path

def linear_regression_prediction():
    """
    Linear Regression 모델을 사용하여 아파트 전월세 수요를 예측하고, 
    결과를 특정 형식의 딕셔너리로 반환하는 함수입니다.
    """
    ##################################################
    # 1. 데이터 불러오기 및 전처리
    ##################################################
    try:
        df = FileUtils().getCsv("아파트(전월세)_실거래가_all.csv")
    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e.filename}")
        return {
            "전세": [],
            "월세": []
        }
   
    df['거래일'] = pd.to_datetime(df['계약년월'].astype(str) + df['계약일'].astype(str).str.zfill(2), format='%Y%m%d')
    bins = [0, 40, 60, 85, 100, 135, 200]
    labels = ['~40㎡', '40~60㎡', '60~85㎡', '85~100㎡', '100~135㎡', '135㎡~']
    df['면적_구간'] = pd.cut(df['전용면적(㎡)'], bins=bins, labels=labels, right=False)
    df['월'] = df['거래일'].dt.to_period('M')
    monthly_demand = df.groupby(['월', '전월세구분', '면적_구간']).size().reset_index(name='수요량')
    monthly_demand['월'] = monthly_demand['월'].dt.to_timestamp()
    monthly_demand_full = monthly_demand.groupby('월').size().reset_index()
    monthly_demand_full['month_index'] = np.arange(len(monthly_demand_full))
    monthly_demand = pd.merge(monthly_demand, monthly_demand_full[['월', 'month_index']], on='월', how='left')

    # 예측할 월 (2025년 10월)의 인덱스 계산
    predict_date_index = (pd.to_datetime('2025-10-01').year - monthly_demand['월'].min().year) * 12 + (pd.to_datetime('2025-10-01').month - monthly_demand['월'].min().month)

    print("--- 면적 구간별 Linear Regression 예측 결과 및 성능 지표 ---")

    # 최종 결과를 담을 딕셔너리
    final_results = {'전세': [], '월세': []}

    for rent_type in ['전세', '월세']:
        print(f"\n==================== {rent_type} ====================")
        for area_label in labels:
            subset_df = monthly_demand[(monthly_demand['전월세구분'] == rent_type) & (monthly_demand['면적_구간'] == area_label)].copy()

            if len(subset_df) < 6:
                print(f"  > {area_label} : 데이터 부족으로 예측 및 평가 불가")
                continue

            X = subset_df[['month_index']]
            y = subset_df['수요량']
            
            # 시계열 데이터 분할 (마지막 3개월을 테스트 데이터로 사용)
            test_size = 3
            X_train = X[:-test_size]
            X_test = X[-test_size:]
            y_train = y[:-test_size]
            y_test = y[-test_size:]
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            predicted_demand_future = model.predict([[predict_date_index]])[0]
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            full_y_pred = np.concatenate([y_train_pred, y_test_pred])

            # 성능 평가
            rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
            r2_train = r2_score(y_train, y_train_pred)
            mae_train = mean_absolute_error(y_train, y_train_pred)
            
            rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
            r2_test = r2_score(y_test, y_test_pred)
            mae_test = mean_absolute_error(y_test, y_test_pred)
            
            overfitting_status = '과적합 의심' if r2_train > r2_test and r2_train - r2_test > 0.2 else '양호'

            print(f"\n--- {area_label} ---")
            print(f"  [Linear Regression] 2025년 10월 예측: {int(predicted_demand_future)}건")
            print(f"  - 훈련 성능: RMSE={rmse_train:.2f}, R²={r2_train:.2f}, MAE={mae_train:.2f}")
            print(f"  - 테스트 성능: RMSE={rmse_test:.2f}, R²={r2_test:.2f}, MAE={mae_test:.2f}")
            print(f"  - 과적합 여부: {overfitting_status}")

            # 예측값과 실제값 차트 시각화 및 저장
            plt.figure(figsize=(10, 6))
            plt.plot(subset_df['월'], y, label='실제 수요량', color='blue', marker='o')
            plt.plot(subset_df['월'], full_y_pred, label='예측 수요량', color='red', linestyle='--')
            plt.title(f'{area_label} {rent_type} - Linear Regression 예측 및 성능')
            plt.xlabel('날짜')
            plt.ylabel('거래량')
            plt.legend()
            plt.grid(True)
            
            image_filename = f"linear_regression_{rent_type}_{area_label}.png"
            image_url = FileUtils().savePngToPath(image_filename)

            # 요청된 형식으로 데이터 구조화
            area_result = {
                area_label: [
                    {"훈련성능": f"RMSE={rmse_train:.2f}, R²={r2_train:.2f}, MAE={mae_train:.2f}"},
                    {"테스트 성능": f"RMSE={rmse_test:.2f}, R²={r2_test:.2f}, MAE={mae_test:.2f}"},
                    {"과적합 여부": overfitting_status},
                    {"실제값 리스트": list(y.astype(int))},
                    {"예측값 리스트": [int(val) for val in full_y_pred]},
                    {"image_url": image_url}
                ]
            }
            final_results[rent_type].append(area_result)
    
    # 최종 2025년 10월 예측 결과 시각화
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('2025년 10월 Linear Regression 최종 예측 (면적 구간별)', fontsize=18)
    
    predicted_demands_jeonse = {label: 0 for label in labels}
    predicted_demands_wolse = {label: 0 for label in labels}

    for result in final_results['전세']:
        for area, data in result.items():
            predicted_demands_jeonse[area] = int(data[4]['예측값 리스트'][-1])

    for result in final_results['월세']:
        for area, data in result.items():
            predicted_demands_wolse[area] = int(data[4]['예측값 리스트'][-1])

    axes[0].bar(labels, [predicted_demands_jeonse[label] for label in labels], color='skyblue')
    axes[0].set_title('전세 수요량', fontsize=15)
    axes[0].set_xlabel('전용면적 (㎡)', fontsize=12)
    axes[0].set_ylabel('예상 거래량', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)

    axes[1].bar(labels, [predicted_demands_wolse[label] for label in labels], color='salmon')
    axes[1].set_title('월세 수요량', fontsize=15)
    axes[1].set_xlabel('전용면적 (㎡)', fontsize=12)
    axes[1].set_ylabel('예상 거래량', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 최종 예측 이미지 저장 및 반환
    final_image_url = FileUtils().savePngToPath("final_linear_regression.png")
    
    return final_results

# Django 뷰에서 이 함수를 호출할 수 있습니다.
# 예를 들어:
# from your_module import linear_regression_prediction
#
# def your_view(request):
#     results = linear_regression_prediction()
#     return JsonResponse(results)