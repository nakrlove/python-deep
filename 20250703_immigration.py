# 파이썬 인터프리터 캐싱 (모듈 캐시)
# import로 모듈을 불러오면 Python은 모듈을 메모리에 캐싱합니다.
# 따라서, 작업중인 소스를 수정하더라도, 이미 import된 상태에서는 반영되지 않습니다.
# 특히 Jupyter, IPython, VSCode Python Interactive 같은 환경에서는 
# 이 캐시가 살아 있어서 재실행 없이 코드 수정만 하면 반영되지 않을 수 있습니다.
# 해결 방법:
# 1) 스크립트를 완전히 종료 후 다시 실행
# 2) 또는 인터프리터 재시작 (VSCode, Jupyter 등에서는 "Restart Python Interpreter")
# 개발 중 임시 대응 (Jupyter/Interactive 용):
# import importlib
# import com_immigration
# importlib.reload(com_immigration)

import importlib
import com_immigration

from com_immigration import ExcelData, PieChart, LineChart ,PlotChart 

importlib.reload(com_immigration)

# TEST2.xlsx파일 Roader
df =  ExcelData()

labels = ['미국', '캐나다', '멕시코', '브라질', '아르헨티나', '칠레', '페루', '콜롬비아', '우루과이', '에콰도르']
LineChart(df,labels)
labels = ['미국', '캐나다', '멕시코', '브라질', '아르헨티나']
# PieChat
PieChart(df,labels)
PlotChart(df)
