import geopandas as gpd

# 원본 GeoJSON 불러오기
gdf = gpd.read_file("20250820/HangJeongDong_ver20250101.geojson")

# 서울특별시만 필터링
seoul_gdf = gdf[gdf["sidonm"] == "서울특별시"]

# '구 단위'로 병합 (sggnm 컬럼 기준)
seoul_gu_union = seoul_gdf.dissolve(by="sggnm")

# 새로운 GeoJSON 파일로 저장
seoul_gu_union.to_file("Seoul_all_gu_boundary.geojson", driver="GeoJSON")