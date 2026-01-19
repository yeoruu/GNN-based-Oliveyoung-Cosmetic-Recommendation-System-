"""
인터랙티브 추천 스크립트
학습된 모델을 로드하여 실시간 추천
"""

import torch
from single_node_gnn_recommender import SingleNodeGNNRecommender


def interactive_recommend():
    """대화형 추천 인터페이스"""
    
    print("="*80)
    print("🌟 화장품 추천 시스템에 오신 것을 환영합니다! 🌟")
    print("="*80)
    
    # 시스템 초기화
    print("\n📚 시스템 초기화 중...")
    recommender = SingleNodeGNNRecommender(
        products_path='final_products.csv',
        reviews_path='final_total_reviews.csv'
    )
    
    # 모델 로드 시도
    try:
        print("\n🔍 학습된 모델을 찾는 중...")
        recommender.model.load_state_dict(
            torch.load('best_single_gnn_model.pt', map_location=recommender.device)
        )
        print("✅ 학습된 모델을 로드했습니다!")
    except:
        print("⚠️  학습된 모델이 없습니다. 새로 학습합니다...")
        recommender.train_model(epochs=30, hidden_channels=128, lr=0.001)
    
    # 카테고리 리스트
    categories = ['전체'] + sorted(recommender.products_df['category'].unique().tolist())
    
    while True:
        print("\n" + "="*80)
        print("🎯 추천 옵션을 선택해주세요")
        print("="*80)
        
        # 1. 피부타입 선택
        print("\n1️⃣  피부타입을 선택하세요:")
        print("   1) 건성")
        print("   2) 지성")
        print("   3) 복합성")
        print("   4) 민감성")
        
        skin_choice = input("\n선택 (1-4): ").strip()
        skin_type_map = {'1': '건성', '2': '지성', '3': '복합성', '4': '민감성'}
        skin_type = skin_type_map.get(skin_choice, '복합성')
        print(f"   ✅ 선택된 피부타입: {skin_type}")
        
        # 2. 카테고리 선택
        print("\n2️⃣  카테고리를 선택하세요:")
        for idx, cat in enumerate(categories, 1):
            print(f"   {idx}) {cat}")
        
        cat_choice = input(f"\n선택 (1-{len(categories)}): ").strip()
        try:
            category_idx = int(cat_choice) - 1
            if 0 <= category_idx < len(categories):
                selected_category = categories[category_idx]
            else:
                selected_category = '전체'
        except:
            selected_category = '전체'
        
        print(f"   ✅ 선택된 카테고리: {selected_category}")
        
        # 3. 선호 제품 선택
        print("\n3️⃣  좋아하는 제품이 있나요?")
        print("   1) 네, 있습니다")
        print("   2) 아니요, 없습니다")
        
        fav_choice = input("\n선택 (1-2): ").strip()
        favorite_product_id = None
        
        if fav_choice == '1':
            # 제품 리스트 보여주기
            print("\n   📋 제품 리스트 (일부):")
            sample_products = recommender.products_df.head(20)
            for idx, row in sample_products.iterrows():
                print(f"      {row['product_id']}: {row['product_name']} ({row['brand']})")
            
            product_id = input("\n   제품 ID를 입력하세요 (예: L1, M1): ").strip().upper()
            if product_id in recommender.products_df['product_id'].values:
                favorite_product_id = product_id
                print(f"   ✅ 선택된 제품: {product_id}")
            else:
                print(f"   ⚠️  제품을 찾을 수 없습니다. 선호 제품 없이 진행합니다.")
        
        # 4. 추천 개수
        print("\n4️⃣  몇 개의 제품을 추천받으시겠어요?")
        num_choice = input("   추천 개수 (1-10, 기본값 5): ").strip()
        try:
            top_n = int(num_choice)
            if top_n < 1 or top_n > 10:
                top_n = 5
        except:
            top_n = 5
        
        print(f"   ✅ {top_n}개 제품을 추천합니다")
        
        # 추천 생성
        print("\n" + "="*80)
        print("🔮 AI가 당신을 위한 최적의 제품을 찾고 있습니다...")
        print("="*80)
        
        recommendations = recommender.recommend(
            skin_type=skin_type,
            category=selected_category if selected_category != '전체' else None,
            favorite_product_id=favorite_product_id,
            top_n=top_n
        )
        
        # 결과 출력
        recommender.print_recommendations(recommendations)
        
        # 계속 진행 여부
        print("\n" + "="*80)
        continue_choice = input("🔄 다시 추천받으시겠어요? (y/n): ").strip().lower()
        if continue_choice != 'y':
            print("\n👋 이용해주셔서 감사합니다!")
            break


def quick_recommend():
    """빠른 추천 (미리 정의된 설정)"""
    
    print("⚡ 빠른 추천 모드")
    print("="*80)
    
    # 시스템 초기화
    recommender = SingleNodeGNNRecommender(
        products_path='final_products.csv',
        reviews_path='final_total_reviews.csv'
    )
    
    # 모델 로드
    try:
        recommender.model.load_state_dict(
            torch.load('best_single_gnn_model.pt', map_location=recommender.device)
        )
    except:
        print("⚠️  학습된 모델이 없습니다. 먼저 main 스크립트를 실행해주세요.")
        return
    
    # 예시 추천들
    test_cases = [
        {
            'name': '건성 피부 로션 추천',
            'skin_type': '건성',
            'category': '로션',
            'favorite': 'L1'
        },
        {
            'name': '지성 피부 전체 카테고리 추천',
            'skin_type': '지성',
            'category': None,
            'favorite': None
        },
        {
            'name': '민감성 피부 세럼 추천',
            'skin_type': '민감성',
            'category': '세럼',
            'favorite': None
        }
    ]
    
    for idx, test in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"📋 테스트 케이스 {idx}: {test['name']}")
        print(f"{'='*80}")
        
        recommendations = recommender.recommend(
            skin_type=test['skin_type'],
            category=test['category'],
            favorite_product_id=test['favorite'],
            top_n=3
        )
        
        recommender.print_recommendations(recommendations)
        
        input("\nEnter를 눌러 다음 테스트로 진행...")


if __name__ == "__main__":
    import sys
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║                                                            ║
    ║          🌟 GNN 기반 화장품 추천 시스템 🌟                  ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    print("실행 모드를 선택하세요:")
    print("  1) 대화형 추천 (interactive)")
    print("  2) 빠른 테스트 (quick test)")
    
    choice = input("\n선택 (1-2): ").strip()
    
    if choice == '2':
        quick_recommend()
    else:
        interactive_recommend()
