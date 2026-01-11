import pandas as pd

# 파일 경로 설정
TABLE2_FILE = "table2_essence_basic.csv"
TABLE3_FILE = "table3_essence_ingredients.csv"
OUTPUT_FILE = "merged_essence_data.csv"

def merge_product_ingredients():
    """
    table2_essence_basic.csv와 table3_essence_ingredients.csv를 병합하여
    product_id, category, brand, product_name, ingredients 컬럼만 추출
    """
    print("="*60)
    print("📋 제품 정보 & 성분 데이터 병합")
    print("="*60)
    
    # 1. table2 (제품 기본 정보) 읽기
    print(f"\n📂 {TABLE2_FILE} 파일 읽는 중...")
    try:
        df_products = pd.read_csv(TABLE2_FILE)
        print(f"   ✅ 로드 완료: {len(df_products)}개 제품")
        print(f"   📊 컬럼: {list(df_products.columns)}")
    except FileNotFoundError:
        print(f"   ❌ 오류: {TABLE2_FILE} 파일을 찾을 수 없습니다.")
        return
    except Exception as e:
        print(f"   ❌ 오류: {str(e)}")
        return
    
    # 2. table3 (성분 정보) 읽기
    print(f"\n📂 {TABLE3_FILE} 파일 읽는 중...")
    try:
        df_ingredients = pd.read_csv(TABLE3_FILE)
        print(f"   ✅ 로드 완료: {len(df_ingredients)}개 제품")
        print(f"   📊 컬럼: {list(df_ingredients.columns)}")
    except FileNotFoundError:
        print(f"   ❌ 오류: {TABLE3_FILE} 파일을 찾을 수 없습니다.")
        return
    except Exception as e:
        print(f"   ❌ 오류: {str(e)}")
        return
    
    # 3. 필요한 컬럼만 선택
    print("\n🔧 필요한 컬럼 추출 중...")
    
    # table2에서: product_id, category, brand, product_name
    if 'product_id' in df_products.columns:
        df_products_selected = df_products[['product_id', 'category', 'brand', 'product_name']].copy()
        print(f"   ✅ table2에서 4개 컬럼 추출 완료")
    else:
        print(f"   ❌ 오류: table2에 'product_id' 컬럼이 없습니다.")
        return
    
    # table3에서: product_id, ingredients
    if 'product_id' in df_ingredients.columns and 'ingredients' in df_ingredients.columns:
        df_ingredients_selected = df_ingredients[['product_id', 'ingredients']].copy()
        print(f"   ✅ table3에서 2개 컬럼 추출 완료")
    else:
        print(f"   ❌ 오류: table3에 필요한 컬럼이 없습니다.")
        print(f"      현재 table3 컬럼: {list(df_ingredients.columns)}")
        return
    
    # 4. product_id를 기준으로 병합 (LEFT JOIN)
    print("\n🔗 데이터 병합 중 (product_id 기준 LEFT JOIN)...")
    df_merged = pd.merge(
        df_products_selected,
        df_ingredients_selected,
        on='product_id',
        how='left'  # table2의 모든 제품 유지
    )
    print(f"   ✅ 병합 완료: {len(df_merged)}개 행")
    
    # 5. 결과 확인
    print("\n📊 병합 결과 통계:")
    print(f"   - 전체 제품 수: {len(df_merged)}개")
    print(f"   - 성분 정보 있는 제품: {df_merged['ingredients'].notna().sum()}개")
    print(f"   - 성분 정보 없는 제품: {df_merged['ingredients'].isna().sum()}개")
    
    # 6. CSV 파일로 저장
    print(f"\n💾 결과 저장 중: {OUTPUT_FILE}")
    try:
        df_merged.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"   ✅ 저장 완료!")
    except Exception as e:
        print(f"   ❌ 저장 실패: {str(e)}")
        return
    
    # 7. 미리보기
    print("\n" + "="*60)
    print("📋 결과 미리보기 (상위 5개)")
    print("="*60)
    print(df_merged.head(5).to_string())
    
    # 8. 컬럼별 null 체크
    print("\n" + "="*60)
    print("📊 컬럼별 결측치 현황")
    print("="*60)
    for col in df_merged.columns:
        null_count = df_merged[col].isna().sum()
        null_pct = (null_count / len(df_merged)) * 100
        print(f"   - {col}: {null_count}개 ({null_pct:.1f}%)")
    
    print("\n" + "="*60)
    print("✅ 완료!")
    print("="*60)
    print(f"출력 파일: {OUTPUT_FILE}")
    print(f"총 {len(df_merged)}개 행, {len(df_merged.columns)}개 컬럼")
    print("="*60)
    
    return df_merged

if __name__ == "__main__":
    # 실행
    df_result = merge_product_ingredients()
    
    # 추가: 성분 정보가 있는 제품만 필터링한 파일도 생성
    if df_result is not None and len(df_result) > 0:
        print("\n💡 추가 작업: 성분 정보가 있는 제품만 필터링...")
        df_with_ingredients = df_result[df_result['ingredients'].notna()].copy()
        
        if len(df_with_ingredients) > 0:
            output_filtered = "merged_essence_data_with_ingredients.csv"
            df_with_ingredients.to_csv(output_filtered, index=False, encoding='utf-8-sig')
            print(f"   ✅ 성분 정보 있는 제품만 저장: {output_filtered}")
            print(f"   📊 총 {len(df_with_ingredients)}개 제품")
        else:
            print("   ⚠️  성분 정보가 있는 제품이 없습니다.")