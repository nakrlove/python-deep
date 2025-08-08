
#문제를 풀어보세요
# diamonds 데이터셋의 'price'를 잘 맞추는 다중선형회귀분석을 해 봅시다.
#(test_set : MSE 가 가장 작아지도록)


import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
url ="https://gist.githubusercontent.com/nnbphuong/def91b5553736764e8e08f6255390f37/raw/373a856a3c9c1119e34b344de9230ae2ea89569d/BostonHousing.csv"
df = pd.read_csv(url)
