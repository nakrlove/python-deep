
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error

# 1. 데이터 불러오기
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target  # 종속변수 (회귀 예시 위해 숫자 사용)

# 2. 훈련/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 교호작용 포함
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = pd.DataFrame(poly.fit_transform(X_train),columns=poly.get_feature_names_out(X_train.columns))
X_test_poly = pd.DataFrame(poly.transform(X_test),columns=poly.get_feature_names_out(X_test.columns))

# 4. 다중공선성 확인 (VIF 계산)
def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [
        variance_inflation_factor(df.values, i) for i in range(df.shape[1])
    ]
    return vif_data

print("=== 초기 VIF ===")
print(calculate_vif(X_train_poly))

# 5. 양방향(stepwise) 변수 선택
def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.05, 
                       threshold_out=0.10, 
                       verbose=True):
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print(f"Add  {best_feature} with p-value {best_pval:.6}")

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        pvalues = model.pvalues.iloc[1:]  # const 제외
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            changed=True
            if verbose:
                print(f"Drop {worst_feature} with p-value {worst_pval:.6}")
        if not changed:
            break
    return included

selected_features = stepwise_selection(X_train_poly, y_train)

print("최종 선택된 변수:", selected_features)

# 6. 최종 모델 학습
X_train_selected = sm.add_constant(X_train_poly[selected_features])
X_test_selected = sm.add_constant(X_test_poly[selected_features])

final_model = sm.OLS(y_train, X_train_selected).fit()
print(final_model.summary())

# 7. MSE 계산
y_pred = final_model.predict(X_test_selected)
mse = mean_squared_error(y_test, y_pred)
print(f"최종 MSE: {mse:.4f}")