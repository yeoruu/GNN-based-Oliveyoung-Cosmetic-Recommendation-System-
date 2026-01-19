"""
개선된 추천 시스템 - 다양성 보장

즉시 적용 가능한 개선:
1. MMR (Maximal Marginal Relevance) - 다양성 보장
2. 카테고리별 제한 - 편향 방지
3. 피부 타입별 점수 조정 - 차별화
4. 성분 기반 보너스 - 맞춤 추천

사용법:
기존 hetero_gnn_enhanced.py 의 HeteroRecommendationSystem 클래스를
이 파일의 ImprovedRecommendationSystem 으로 교체
"""

import numpy as np
import torch
from collections import defaultdict


class ImprovedRecommendationSystem:
    """개선된 추천 시스템 - 다양성 보장"""
    
    def __init__(self, model, hetero_data, data_loader):
        self.model = model
        self.hetero_data = hetero_data
        self.data_loader = data_loader
        
        # 피부 타입별 선호 성분
        self.skintype_ingredients = {
            '건성': ['세라마이드', '히알루론산', '스쿠알란', '글리세린', '세라믹', '판테놀'],
            '지성': ['티트리', '살리실산', '나이아신아마이드', 'BHA', '녹차', '아연'],
            '민감성': ['센텔라', '알로에', '판테놀', '알란토인', '시카', '마데카소사이드'],
            '복합성': ['나이아신아마이드', '녹차', '티트리', '히알루론산', '세라마이드'],
            '여드름성': ['티트리', '살리실산', 'AHA', 'BHA', '징크', '프로폴리스'],
            '아토피': ['세라마이드', '판테놀', '알란토인', '콜로이달오트밀', '시카'],
            '트러블': ['센텔라', '티트리', '프로폴리스', '알란토인', '아연']
        }
        
        # 카테고리 우선순위 (다양성 확보)
        self.category_priority = [
            '크림', '세럼/에센스/앰플', '로션', '토너', '오일', 
            '미스트', '팩/마스크', '클렌저', '선케어'
        ]
    
    def recommend_by_skintype(self, skintype, top_k=10, diversity_weight=0.3):
        """다양성을 고려한 피부 타입별 추천"""
        
        if skintype not in self.data_loader.skintype_to_idx:
            print(f"❌ SkinType '{skintype}' not found!")
            return []
        
        # 1. 기본 점수 계산 (많은 후보)
        candidate_size = min(100, len(self.data_loader.product_to_idx))
        candidates = self._get_base_scores(skintype, top_n=candidate_size)
        
        # 2. 성분 기반 점수 조정
        candidates = self._adjust_by_ingredients(candidates, skintype)
        
        # 3. MMR로 다양성 보장하며 선택
        recommendations = self._select_with_mmr(
            candidates, 
            top_k=top_k,
            diversity_weight=diversity_weight
        )
        
        return recommendations
    
    def _get_base_scores(self, skintype, top_n=100):
        """기본 점수 계산"""
        skintype_idx = self.data_loader.skintype_to_idx[skintype]
        
        self.model.eval()
        with torch.no_grad():
            num_products = len(self.data_loader.product_to_idx)
            device = next(self.model.parameters()).device
            
            product_indices = torch.arange(num_products, dtype=torch.long).to(device)
            skintype_indices = torch.full((num_products,), skintype_idx, dtype=torch.long).to(device)
            
            scores = self.model(self.hetero_data, skintype_indices, product_indices)
            scores = scores.cpu().numpy()
        
        # 상위 N개 후보
        top_indices = np.argsort(scores)[-top_n:][::-1]
        
        candidates = []
        for idx in top_indices:
            product_id = self.data_loader.idx_to_product[idx]
            product = self.data_loader.products_df[
                self.data_loader.products_df['product_id'] == product_id
            ].iloc[0]
            
            candidates.append({
                'product_idx': idx,
                'product_id': product_id,
                'product_name': product['product_name'],
                'brand': product['brand'],
                'category': product['category'],
                'base_score': float(scores[idx]),
                'adjusted_score': float(scores[idx]),
                'ingredients': product['ingredient_list']
            })
        
        return candidates
    
    def _adjust_by_ingredients(self, candidates, skintype):
        """성분 기반 점수 조정"""
        preferred_ings = self.skintype_ingredients.get(skintype, [])
        
        for candidate in candidates:
            bonus = 0.0
            matched_ingredients = []
            
            # 선호 성분 매칭
            for pref_ing in preferred_ings:
                for product_ing in candidate['ingredients']:
                    # 대소문자 무시, 부분 매칭
                    if pref_ing.lower() in product_ing.lower():
                        bonus += 0.05  # 성분당 0.05점 보너스
                        matched_ingredients.append(pref_ing)
                        break  # 중복 카운트 방지
            
            # 최대 보너스 0.3점
            bonus = min(bonus, 0.3)
            
            candidate['adjusted_score'] += bonus
            candidate['matched_ingredients'] = matched_ingredients
        
        # 조정된 점수로 재정렬
        candidates.sort(key=lambda x: x['adjusted_score'], reverse=True)
        
        return candidates
    
    def _select_with_mmr(self, candidates, top_k=10, diversity_weight=0.3):
        """MMR (Maximal Marginal Relevance)로 다양성 보장하며 선택"""
        
        selected = []
        selected_categories = []
        selected_brands = []
        category_count = defaultdict(int)
        brand_count = defaultdict(int)
        
        # 카테고리별 최대 개수
        max_per_category = max(2, top_k // 3)  # 최소 2개, 또는 전체의 1/3
        max_per_brand = max(2, top_k // 4)     # 최소 2개, 또는 전체의 1/4
        
        for _ in range(top_k):
            if not candidates:
                break
            
            best_score = -float('inf')
            best_idx = -1
            best_candidate = None
            
            for idx, candidate in enumerate(candidates):
                if candidate in selected:
                    continue
                
                # 1. 관련성 점수 (adjusted_score)
                relevance = candidate['adjusted_score']
                
                # 2. 다양성 점수
                diversity = self._calculate_diversity(
                    candidate,
                    selected,
                    selected_categories,
                    selected_brands,
                    category_count,
                    brand_count,
                    max_per_category,
                    max_per_brand
                )
                
                # 3. MMR 점수
                mmr_score = (1 - diversity_weight) * relevance + diversity_weight * diversity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                selected_categories.append(best_candidate['category'])
                selected_brands.append(best_candidate['brand'])
                category_count[best_candidate['category']] += 1
                brand_count[best_candidate['brand']] += 1
                
                # 선택된 후보 제거
                candidates.pop(best_idx)
        
        # 최종 형식으로 변환
        recommendations = []
        for rank, item in enumerate(selected, 1):
            recommendations.append({
                'rank': rank,
                'product_id': item['product_id'],
                'product_name': item['product_name'],
                'brand': item['brand'],
                'category': item['category'],
                'predicted_rating': item['adjusted_score'],
                'base_rating': item['base_score'],
                'ingredients': item['ingredients'][:5],
                'matched_ingredients': item.get('matched_ingredients', [])
            })
        
        return recommendations
    
    def _calculate_diversity(self, candidate, selected, selected_categories, 
                            selected_brands, category_count, brand_count,
                            max_per_category, max_per_brand):
        """다양성 점수 계산"""
        
        if not selected:
            return 1.0
        
        diversity_score = 0.0
        
        # 1. 카테고리 다양성 (가중치: 0.5)
        category = candidate['category']
        
        if category_count[category] >= max_per_category:
            # 이미 많으면 큰 페널티
            category_diversity = 0.0
        elif category in selected_categories:
            # 있지만 제한 안 넘었으면 작은 페널티
            category_diversity = 0.3
        else:
            # 새로운 카테고리면 보너스
            category_diversity = 1.0
        
        diversity_score += 0.5 * category_diversity
        
        # 2. 브랜드 다양성 (가중치: 0.3)
        brand = candidate['brand']
        
        if brand_count[brand] >= max_per_brand:
            brand_diversity = 0.0
        elif brand in selected_brands:
            brand_diversity = 0.5
        else:
            brand_diversity = 1.0
        
        diversity_score += 0.3 * brand_diversity
        
        # 3. 성분 다양성 (가중치: 0.2)
        candidate_ings = set(candidate['ingredients'])
        
        # 이미 선택된 제품들과 성분 유사도 계산
        avg_similarity = 0.0
        for sel in selected:
            sel_ings = set(sel['ingredients'])
            similarity = len(candidate_ings & sel_ings) / len(candidate_ings | sel_ings) if candidate_ings or sel_ings else 0
            avg_similarity += similarity
        
        if selected:
            avg_similarity /= len(selected)
        
        ingredient_diversity = 1.0 - avg_similarity
        diversity_score += 0.2 * ingredient_diversity
        
        return diversity_score
    
    def recommend_comparison(self, skintypes, top_k=5):
        """여러 피부 타입 비교 추천"""
        
        print("=" * 80)
        print("피부 타입별 추천 비교 (개선 버전)".center(80))
        print("=" * 80)
        
        for skintype in skintypes:
            print(f"\n👤 피부 타입: {skintype}")
            print("-" * 80)
            
            recommendations = self.recommend_by_skintype(skintype, top_k=top_k)
            
            for rec in recommendations:
                print(f"\n{rec['rank']}. {rec['product_name']}")
                print(f"   브랜드: {rec['brand']} | 카테고리: {rec['category']}")
                print(f"   예상 평점: {rec['predicted_rating']:.2f}/5.0 (기본: {rec['base_rating']:.2f})")
                
                # 매칭된 성분 표시
                if rec.get('matched_ingredients'):
                    print(f"   ✨ 매칭 성분: {', '.join(rec['matched_ingredients'][:3])}")
                
                print(f"   주요 성분: {', '.join(rec['ingredients'][:3])}")
        
        print("\n" + "=" * 80)


# ============================================================================
# 사용 예시
# ============================================================================

def demo_improved_recommendation(model, hetero_data, data_loader):
    """개선된 추천 시스템 데모"""
    
    print("\n" + "=" * 80)
    print("🎯 개선된 추천 시스템 실행".center(80))
    print("=" * 80)
    
    # 개선된 추천 시스템 생성
    rec_system = ImprovedRecommendationSystem(model, hetero_data, data_loader)
    
    # 피부 타입 선택
    sample_skintypes = ['건성', '민감성', '복합성']
    
    # 비교 추천
    rec_system.recommend_comparison(sample_skintypes, top_k=5)
    
    # 통계 출력
    print("\n📊 추천 통계:")
    
    for skintype in sample_skintypes:
        recommendations = rec_system.recommend_by_skintype(skintype, top_k=10)
        
        categories = [rec['category'] for rec in recommendations]
        brands = [rec['brand'] for rec in recommendations]
        
        unique_categories = len(set(categories))
        unique_brands = len(set(brands))
        
        score_range = max(rec['predicted_rating'] for rec in recommendations) - \
                     min(rec['predicted_rating'] for rec in recommendations)
        
        print(f"\n{skintype}:")
        print(f"   • 카테고리 다양성: {unique_categories}/10")
        print(f"   • 브랜드 다양성: {unique_brands}/10")
        print(f"   • 점수 범위: {score_range:.2f}점")
    
    return rec_system


# ============================================================================
# 메인 함수에 추가
# ============================================================================

"""
hetero_gnn_enhanced.py 의 main() 함수에서:

# 기존 코드:
rec_system = HeteroRecommendationSystem(model, hetero_data, data_loader)

# 변경:
rec_system = ImprovedRecommendationSystem(model, hetero_data, data_loader)

# 또는 비교:
print("\n=== 기존 추천 시스템 ===")
old_rec_system = HeteroRecommendationSystem(model, hetero_data, data_loader)
# ... 추천 ...

print("\n=== 개선된 추천 시스템 ===")
new_rec_system = ImprovedRecommendationSystem(model, hetero_data, data_loader)
# ... 추천 ...
"""