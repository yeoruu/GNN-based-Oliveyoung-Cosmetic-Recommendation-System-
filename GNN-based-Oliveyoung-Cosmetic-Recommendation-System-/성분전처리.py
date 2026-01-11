import pandas as pd
import re
from collections import Counter
import numpy as np
from tqdm import tqdm

class CosmeticIngredientPreprocessor:
    """올리브영 화장품 성분 전처리 클래스"""
    
    def __init__(self):
        # 동의어 사전
        self.synonym_dict = {
            # 물 관련
            '정제수': 'water',
            'purified water': 'water',
            '물': 'water',
            
            # 히알루론산 관련
            '히알루론산': 'hyaluronic_acid',
            '히알루론산나트륨': 'hyaluronic_acid',
            'sodium hyaluronate': 'hyaluronic_acid',
            'hyaluronic acid': 'hyaluronic_acid',
            '소듐하이알루로네이트': 'hyaluronic_acid',
            
            # 글리세린 관련
            '글리세린': 'glycerin',
            'glycerin': 'glycerin',
            'glycerine': 'glycerin',
            
            # 나이아신아마이드 관련
            '나이아신아마이드': 'niacinamide',
            'niacinamide': 'niacinamide',
            
            # 부틸렌글라이콜/글리콜 관련
            '부틸렌글라이콜': 'butylene_glycol',
            '부틸렌글리콜': 'butylene_glycol',
            'butylene glycol': 'butylene_glycol',
            
            # 세라마이드 관련
            '세라마이드': 'ceramide',
            'ceramide': 'ceramide',
            
            # 판테놀 관련
            '판테놀': 'panthenol',
            'panthenol': 'panthenol',
            'd-판테놀': 'panthenol',
            '덱스판테놀': 'panthenol',
            
            # 알란토인 관련
            '알란토인': 'allantoin',
            'allantoin': 'allantoin',
            
            # 레티놀 관련
            '레티놀': 'retinol',
            'retinol': 'retinol',
            
            # 비타민C 관련
            '아스코르브산': 'vitamin_c',
            'ascorbic acid': 'vitamin_c',
            '비타민c': 'vitamin_c',
            
            # 센텔라 관련
            '센텔라아시아티카추출물': 'centella',
            'centella asiatica extract': 'centella',
            '병풀추출물': 'centella',
            
            # 프로폴리스 관련
            '프로폴리스추출물': 'propolis',
            'propolis extract': 'propolis',
            '프로폴리스': 'propolis',
            
            # 카프릴릭/카프릭 관련
            '카프릴릭/카프릭트라이글리세라이드': 'caprylic_capric_triglyceride',
            '카프릴릭카프릭트라이글리세라이드': 'caprylic_capric_triglyceride',
            
            # 프로판다이올
            '프로판다이올': 'propanediol',
            'propanediol': 'propanediol',
        }
        
        # 불필요한 성분 (너무 일반적)
        self.stopwords = {
            'water', '정제수', '물',
        }
        
        # 유해/주의 성분
        self.harmful_ingredients = {
            'paraben', '파라벤', 
            'sls', 'sodium lauryl sulfate',
            'sles', 'sodium laureth sulfate',
            'mineral oil', '미네랄오일',
            'alcohol denat', '변성알코올',
            'fragrance', '향료'
        }
    
    def parse_ingredient_string(self, ingredient_text):
        """
        성분 문자열 파싱
        - 쉼표로 구분
        - 괄호 안 내용 처리
        """
        if pd.isna(ingredient_text) or ingredient_text == '' or not isinstance(ingredient_text, str):
            return []
        
        # 쉼표로 분리 (단, 괄호 안은 제외)
        # 예: "성분A(100ml, 50%), 성분B" -> ["성분A(100ml, 50%)", "성분B"]
        parts = []
        current = ""
        depth = 0
        
        for char in ingredient_text:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            elif char == ',' and depth == 0:
                if current.strip():
                    parts.append(current.strip())
                current = ""
                continue
            current += char
        
        if current.strip():
            parts.append(current.strip())
        
        return parts
    
    def clean_ingredient(self, ingredient):
        """
        개별 성분 정제
        """
        if not ingredient:
            return ""
        
        # 소문자 변환
        ingredient = ingredient.lower()
        
        # 괄호와 괄호 안 내용 제거
        # "정제수(purified water)" → "정제수"
        # "나이아신아마이드(1%)" → "나이아신아마이드"
        ingredient = re.sub(r'\([^)]*\)', '', ingredient)
        ingredient = re.sub(r'\[[^\]]*\]', '', ingredient)
        
        # 농도 정보 제거: "2%", "100ml" 등
        ingredient = re.sub(r'\d+\.?\d*\s*(%|ml|mg|g|ppm)', '', ingredient)
        
        # 특수문자 정리 (하이픈, 슬래시는 언더스코어로 변환)
        ingredient = ingredient.replace('/', '_')
        ingredient = re.sub(r'[^\w\s\-_가-힣]', ' ', ingredient)
        
        # 여러 공백을 하나로
        ingredient = re.sub(r'\s+', ' ', ingredient)
        
        # 앞뒤 공백 제거
        ingredient = ingredient.strip()
        
        return ingredient
    
    def normalize_ingredient(self, ingredient):
        """
        성분 표준화 (동의어 통일)
        """
        if not ingredient:
            return ""
        
        ingredient = ingredient.lower()
        
        # 정확히 일치하는 경우
        if ingredient in self.synonym_dict:
            return self.synonym_dict[ingredient]
        
        # 부분 매칭 (더 긴 키워드부터 매칭)
        sorted_keys = sorted(self.synonym_dict.keys(), key=len, reverse=True)
        for key in sorted_keys:
            if key in ingredient:
                return self.synonym_dict[key]
        
        return ingredient
    
    def filter_ingredients(self, ingredients, top_n=None, remove_stopwords=True):
        """
        성분 필터링
        """
        filtered = []
        
        for ing in ingredients:
            if not ing:
                continue
            
            # stopwords 제거
            if remove_stopwords and ing in self.stopwords:
                continue
            
            filtered.append(ing)
            
            # 상위 N개만 유지 (농도 순서 고려)
            if top_n and len(filtered) >= top_n:
                break
        
        return filtered
    
    def identify_harmful_ingredients(self, ingredients):
        """
        유해 성분 식별
        """
        harmful = []
        for ing in ingredients:
            for harmful_ing in self.harmful_ingredients:
                if harmful_ing in ing.lower():
                    harmful.append(ing)
                    break
        return harmful
    
    def process_ingredients(self, ingredient_text, top_n=20, remove_stopwords=True):
        """
        전체 전처리 파이프라인
        """
        # 1. 파싱
        ingredients = self.parse_ingredient_string(ingredient_text)
        
        # 2. 정제
        ingredients = [self.clean_ingredient(ing) for ing in ingredients]
        
        # 3. 빈 문자열 제거
        ingredients = [ing for ing in ingredients if ing]
        
        # 4. 표준화
        ingredients = [self.normalize_ingredient(ing) for ing in ingredients]
        
        # 5. 필터링
        ingredients = self.filter_ingredients(ingredients, top_n, remove_stopwords)
        
        # 6. 중복 제거 (순서 유지)
        seen = set()
        unique_ingredients = []
        for ing in ingredients:
            if ing and ing not in seen:
                seen.add(ing)
                unique_ingredients.append(ing)
        
        return unique_ingredients
    
    def get_ingredient_stats(self, df, ingredient_col='ingredients'):
        """
        성분 통계 분석
        """
        all_ingredients = []
        
        print("성분 통계 분석 중...")
        for ing_text in tqdm(df[ingredient_col]):
            processed = self.process_ingredients(ing_text)
            all_ingredients.extend(processed)
        
        # 빈도 계산
        ingredient_counts = Counter(all_ingredients)
        
        stats = {
            'total_unique_ingredients': len(ingredient_counts),
            'most_common': ingredient_counts.most_common(30),
            'avg_ingredients_per_product': len(all_ingredients) / len(df) if len(df) > 0 else 0,
            'total_ingredients': len(all_ingredients)
        }
        
        return stats, ingredient_counts


def preprocess_oliveyoung_data(csv_path):
    """
    올리브영 CSV 데이터 전처리
    """
    print(f"데이터 로딩 중: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"총 제품 수: {len(df)}")
    print(f"컬럼: {df.columns.tolist()}")
    
    # 성분 컬럼 확인 (실제 컬럼명에 맞게 조정)
    ingredient_col = None
    for col in df.columns:
        if 'ingredient' in col.lower() or '성분' in col:
            ingredient_col = col
            break
    
    if ingredient_col is None:
        print("⚠️ 성분 컬럼을 찾을 수 없습니다. 컬럼명을 확인하세요.")
        print(f"사용 가능한 컬럼: {df.columns.tolist()}")
        return df
    
    print(f"✅ 성분 컬럼 발견: '{ingredient_col}'")
    
    # 전처리 시작
    preprocessor = CosmeticIngredientPreprocessor()
    
    print("\n성분 전처리 중...")
    df['processed_ingredients'] = df[ingredient_col].apply(
        lambda x: preprocessor.process_ingredients(x, top_n=20)
    )
    
    print("유해 성분 식별 중...")
    df['harmful_ingredients'] = df['processed_ingredients'].apply(
        preprocessor.identify_harmful_ingredients
    )
    
    print("기타 특징 추출 중...")
    # 성분 개수
    df['num_ingredients'] = df['processed_ingredients'].apply(len)
    
    # 주요 성분 (상위 5개)
    df['key_ingredients'] = df['processed_ingredients'].apply(
        lambda x: x[:5] if len(x) >= 5 else x
    )
    
    # 성분을 문자열로 변환 (저장용)
    df['ingredients_str'] = df['processed_ingredients'].apply(lambda x: '|'.join(x))
    
    # 통계 출력
    print("\n" + "="*60)
    print("전처리 통계")
    print("="*60)
    stats, ingredient_counts = preprocessor.get_ingredient_stats(df, ingredient_col)
    
    print(f"총 고유 성분 수: {stats['total_unique_ingredients']}")
    print(f"총 성분 출현 횟수: {stats['total_ingredients']}")
    print(f"제품당 평균 성분 수: {stats['avg_ingredients_per_product']:.2f}")
    
    print(f"\n가장 많이 사용된 성분 Top 20:")
    for i, (ing, count) in enumerate(stats['most_common'][:20], 1):
        print(f"  {i:2d}. {ing:40s} : {count:4d}회")
    
    # 유해 성분이 있는 제품 비율
    harmful_products = df[df['harmful_ingredients'].apply(len) > 0]
    print(f"\n⚠️ 유해 성분 포함 제품: {len(harmful_products)}개 ({len(harmful_products)/len(df)*100:.1f}%)")
    
    return df, ingredient_counts, preprocessor


def save_processed_data(df, output_path):
    """
    전처리된 데이터 저장
    """
    # 리스트 타입 컬럼을 문자열로 변환
    df_to_save = df.copy()
    
    list_columns = ['processed_ingredients', 'harmful_ingredients', 'key_ingredients']
    for col in list_columns:
        if col in df_to_save.columns:
            df_to_save[col] = df_to_save[col].apply(lambda x: '|'.join(x) if isinstance(x, list) else x)
    
    df_to_save.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 전처리 완료! 저장 경로: {output_path}")


def analyze_specific_products(df, n_samples=5):
    """
    샘플 제품 분석
    """
    print("\n" + "="*60)
    print(f"샘플 제품 분석 (무작위 {n_samples}개)")
    print("="*60)
    
    samples = df.sample(min(n_samples, len(df)))
    
    for idx, row in samples.iterrows():
        print(f"\n[제품 {idx}]")
        if 'product_name' in df.columns:
            print(f"제품명: {row['product_name']}")
        elif '제품명' in df.columns:
            print(f"제품명: {row['제품명']}")
        
        print(f"성분 개수: {row['num_ingredients']}")
        print(f"주요 성분 (Top 5): {', '.join(row['key_ingredients'])}")
        
        if row['harmful_ingredients']:
            print(f"⚠️ 유해 성분: {', '.join(row['harmful_ingredients'])}")
        else:
            print("✅ 유해 성분 없음")
        
        print(f"전체 성분: {', '.join(row['processed_ingredients'][:10])}...")


# ==================== 실행 ====================

if __name__ == "__main__":
    # CSV 파일 경로
    csv_path = "merged_essence_data.csv"
    
    # 전처리 실행
    df_processed, ingredient_counts, preprocessor = preprocess_oliveyoung_data(csv_path)
    
    # 샘플 제품 분석
    analyze_specific_products(df_processed, n_samples=5)
    
    # 결과 저장
    output_path = "merged_essence_data_preprocessed.csv"
    save_processed_data(df_processed, output_path)
    
    print("\n" + "="*60)
    print("✨ 전처리 완료!")
    print("="*60)