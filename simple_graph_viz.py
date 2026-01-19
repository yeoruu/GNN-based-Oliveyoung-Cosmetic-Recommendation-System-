"""
간단한 그래프 시각화 (의존성 최소)

필수 라이브러리:
pip install networkx matplotlib pandas

실행:
python simple_graph_viz.py
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import ast

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # Mac
plt.rcParams['axes.unicode_minus'] = False

print("간단한 그래프 시각화")
print("=" * 60)

# 데이터 로드
print("\n📂 데이터 로드 중...")
products_df = pd.read_csv('merge_final/final_products.csv')
reviews_df = pd.read_csv('merge_final/final_total_reviews.csv')

# 성분 파싱
def parse_ingredients(ing_str):
    if pd.isna(ing_str):
        return []
    try:
        return ast.literal_eval(ing_str) if isinstance(ing_str, str) else []
    except:
        return []

products_df['ingredient_list'] = products_df['ingredients'].apply(parse_ingredients)

# 피부 타입 파싱
def parse_skintype(st):
    if pd.isna(st):
        return []
    for sep in ['|', ',', '/']:
        if sep in str(st):
            return [s.strip() for s in str(st).split(sep)]
    return [str(st).strip()]

reviews_df['skintype_list'] = reviews_df['user_keywords'].apply(parse_skintype)

# 샘플 선택
sample_products = products_df.head(3)
print(f"   ✓ 제품 3개 선택")

# 그래프 생성
G = nx.Graph()

# 노드 색상
colors = []
labels = {}

# 1. 제품 노드 추가
for idx, product in sample_products.iterrows():
    node_id = f"제품{idx}"
    G.add_node(node_id)
    colors.append('#FF6B6B')  # 빨강
    labels[node_id] = product['product_name'][:10]

# 2. 피부 타입 노드
all_skintypes = set()
for types in reviews_df['skintype_list']:
    all_skintypes.update(types)

for st in list(all_skintypes)[:3]:
    G.add_node(st)
    colors.append('#4ECDC4')  # 청록
    labels[st] = st

# 3. 성분 노드 (상위 5개)
top_ingredients = []
for ings in sample_products['ingredient_list']:
    top_ingredients.extend(ings[:2])

for ing in list(set(top_ingredients))[:5]:
    G.add_node(ing)
    colors.append('#95E1D3')  # 연한 청록
    labels[ing] = ing[:6]

# 4. 카테고리 노드
for cat in sample_products['category'].unique():
    G.add_node(cat)
    colors.append('#FFE66D')  # 노랑
    labels[cat] = cat

# 엣지 추가
print("\n🔗 엣지 생성 중...")

# 제품 - 성분
for idx, product in sample_products.iterrows():
    product_node = f"제품{idx}"
    for ing in product['ingredient_list'][:2]:
        if ing in G.nodes():
            G.add_edge(product_node, ing)

# 제품 - 카테고리
for idx, product in sample_products.iterrows():
    product_node = f"제품{idx}"
    cat = product['category']
    if cat in G.nodes():
        G.add_edge(product_node, cat)

# 피부타입 - 제품
for _, review in reviews_df.head(20).iterrows():
    if review['product_id'] not in sample_products['product_id'].values:
        continue
    
    prod_idx = sample_products[sample_products['product_id'] == review['product_id']].index[0]
    product_node = f"제품{prod_idx}"
    
    for st in review['skintype_list']:
        if st in G.nodes():
            G.add_edge(st, product_node)

print(f"   ✓ 노드: {G.number_of_nodes()}개")
print(f"   ✓ 엣지: {G.number_of_edges()}개")

# 시각화
print("\n🎨 시각화 생성 중...")

plt.figure(figsize=(16, 12))

pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

# 노드 그리기
nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=2000, alpha=0.9)

# 엣지 그리기
nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, edge_color='gray')

# 레이블
nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')

plt.title("이종 그래프 구조 (샘플)", fontsize=18, fontweight='bold', pad=20)
plt.axis('off')
plt.tight_layout()

# 저장
output_file = 'simple_graph_viz.png'
plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')

print(f"   ✓ 저장: {output_file}")
print("\n✅ 완료!")
print(f"\n📁 {output_file} 파일을 확인하세요!")

# 통계 출력
print("\n📊 그래프 통계:")
print(f"   노드 타입:")
print(f"   • 제품: 빨강")
print(f"   • 피부타입: 청록")
print(f"   • 성분: 연한 청록")
print(f"   • 카테고리: 노랑")

plt.show()
