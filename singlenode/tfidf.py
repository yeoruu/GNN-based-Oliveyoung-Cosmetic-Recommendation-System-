"""
단일 노드 GNN 기반 화장품 추천시스템
- 제품 노드 피처: 카테고리 + 브랜드 + 성분(TF-IDF->SVD) + 피부타입별 평균평점
- 엣지: 성분(TF-IDF->SVD) 코사인 유사도 기반 + 기타 규칙
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import roc_auc_score, average_precision_score, ndcg_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import ast
import warnings
warnings.filterwarnings('ignore')


# =========================
# 모델
# =========================
class ProductGNN(nn.Module):
    """단일 노드 (제품) GNN 모델"""
    def __init__(self, in_channels, hidden_channels, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # 사용자 피처 차원 (피부타입 7개 + 평점 1개 = 8)
        user_feature_dim = 8
        
        combined_dim = user_feature_dim + hidden_channels
        self.predictor = nn.Sequential(
            nn.Linear(combined_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, 1)
        )
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.3, training=self.training)
        return x
    
    def predict_rating(self, x, edge_index, user_features, product_indices):
        """
        user_features: [B,8] or [8] or [1,8]
        product_indices: [B] (or 전체 제품 인덱스)
        """
        product_emb = self.forward(x, edge_index)
        
        if user_features.dim() == 1:
            user_emb = user_features.unsqueeze(0).repeat(len(product_indices), 1)
        else:
            if user_features.size(0) == 1 and len(product_indices) > 1:
                user_emb = user_features.repeat(len(product_indices), 1)
            else:
                user_emb = user_features
        
        selected_product_emb = product_emb[product_indices]
        combined = torch.cat([user_emb, selected_product_emb], dim=-1)
        return self.predictor(combined).squeeze()


# =========================
# 추천 시스템
# =========================
class SingleNodeGNNRecommender:
    # ✅ 데이터 카테고리 5개로 고정
    FIXED_CATEGORIES = ["에센스/세럼/앰플", "크림", "스킨/토너", "로션", "미스트/오일"]

    def __init__(self, products_path, reviews_path, svd_dim=100, debug=False):
        print("📚 데이터 로딩 중...")
        self.products_df = pd.read_csv(products_path)
        self.reviews_df = pd.read_csv(reviews_path, encoding='utf-8-sig')
        self.debug = debug

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  디바이스: {self.device}")

        # 인코더
        self.product_encoder = LabelEncoder()
        self.category_encoder = LabelEncoder()
        self.brand_encoder = LabelEncoder()
        self.mlb_ingredients = MultiLabelBinarizer()
        self.user_encoder = LabelEncoder()

        # TF-IDF SVD 차원
        self.svd_dim = svd_dim

        self._preprocess_data()
        self._build_graph()

        # ✅ 데이터에 실제로 존재하는 카테고리 검증 (안 맞으면 바로 터뜨리기)
        real_cats = set(self.products_df["category"].astype(str).str.strip().unique())
        missing = [c for c in self.FIXED_CATEGORIES if c not in real_cats]
        if missing:
            raise ValueError(f"FIXED_CATEGORIES 중 데이터에 없는 값이 있음: {missing} | 실제 카테고리: {sorted(list(real_cats))}")

    # ---------- 카테고리 검증/입력 ----------
    def validate_category(self, category: str):
        if category not in self.FIXED_CATEGORIES:
            raise ValueError(f"카테고리는 5개 중 하나여야 함: {self.FIXED_CATEGORIES} | 입력: {category}")

    def choose_category_cli(self):
        print("\n카테고리 선택 (1~5):")
        for i, c in enumerate(self.FIXED_CATEGORIES, 1):
            print(f"  {i}) {c}")
        while True:
            raw = input("번호 입력: ").strip()
            if raw.isdigit():
                idx = int(raw)
                if 1 <= idx <= len(self.FIXED_CATEGORIES):
                    return self.FIXED_CATEGORIES[idx - 1]
            print("❌ 잘못 입력함. 1~5 중에서 골라줘.")

    # ---------- 전처리 ----------
    def _preprocess_data(self):
        print("🔧 데이터 전처리 중...")

        def parse_ingredients(x):
            if not isinstance(x, str):
                return []
            try:
                x = x.replace("’", "'").replace("‘", "'")
                x = x.replace('“', '"').replace('”', '"')
                return ast.literal_eval(x)
            except:
                return []

        self.products_df['ingredients_list'] = self.products_df['ingredients'].apply(parse_ingredients)

        # 리뷰 스킨타입 결측 처리
        if self.reviews_df['user_keywords'].isna().any():
            na_count = self.reviews_df['user_keywords'].isna().sum()
            print(f"⚠️  피부타입 결측값 {na_count}개를 '알 수 없음'으로 처리합니다.")
            self.reviews_df['user_keywords'] = self.reviews_df['user_keywords'].fillna('알 수 없음')

        self.reviews_df['skin_types'] = self.reviews_df['user_keywords'].apply(
            lambda x: [t.strip() for t in str(x).split('|')]
        )

        self.reviews_df['rating_normalized'] = self.reviews_df['user_rating'] / 5.0

        # 인코딩
        self.products_df['product_encoded'] = self.product_encoder.fit_transform(self.products_df['product_id'])
        self.products_df['category_encoded'] = self.category_encoder.fit_transform(self.products_df['category'])
        self.products_df['brand_encoded'] = self.brand_encoder.fit_transform(self.products_df['brand'])
        self.reviews_df['user_encoded'] = self.user_encoder.fit_transform(self.reviews_df['user_id'])

        # 리뷰에 제품 인코딩 매핑
        product_to_encoded = dict(zip(self.products_df['product_id'], self.products_df['product_encoded']))
        self.reviews_df['product_encoded'] = self.reviews_df['product_id'].map(product_to_encoded)

        # (참고용) 성분 vocab 수
        all_ingredients = self.products_df['ingredients_list'].tolist()
        self.ingredient_matrix = self.mlb_ingredients.fit_transform(all_ingredients)

        # ✅ 성분 TF-IDF -> SVD 임베딩
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            token_pattern=None,
            lowercase=False
        )
        self.ingredient_tfidf = self.tfidf_vectorizer.fit_transform(self.products_df['ingredients_list'])

        self.svd = TruncatedSVD(n_components=self.svd_dim, random_state=42)
        ingredient_svd = self.svd.fit_transform(self.ingredient_tfidf)     # (N, svd_dim)
        ingredient_svd = normalize(ingredient_svd)                         # L2 normalize
        self.ingredient_tfidf_svd = ingredient_svd                         # numpy (N, svd_dim)

        print(f"✅ 제품 수: {len(self.product_encoder.classes_)}")
        print(f"✅ 사용자 수: {len(self.user_encoder.classes_)}")
        print(f"✅ 리뷰 수: {len(self.reviews_df)}")
        print(f"✅ 카테고리 수: {len(self.category_encoder.classes_)}")
        print(f"✅ 브랜드 수: {len(self.brand_encoder.classes_)}")
        print(f"✅ 성분 수: {self.ingredient_matrix.shape[1]}")
        print(f"✅ 성분 TF-IDF shape: {self.ingredient_tfidf.shape}")
        print(f"✅ 성분 TF-IDF(SVD) shape: {self.ingredient_tfidf_svd.shape}")

    # ---------- 그래프 ----------
    def _build_graph(self):
        print("🔨 그래프 구축 중...")

        # 카테고리 one-hot
        num_categories = len(self.category_encoder.classes_)
        category_onehot = np.eye(num_categories)[self.products_df['category_encoded']]

        # 브랜드 one-hot
        num_brands = len(self.brand_encoder.classes_)
        brand_onehot = np.eye(num_brands)[self.products_df['brand_encoded']]

        # 피부타입별 평균 평점
        skin_types_all = ['건성', '민감성', '복합성', '약건성', '지성', '트러블성', '중성']
        skin_ratings = np.zeros((len(self.products_df), len(skin_types_all)))

        for idx, product_id in enumerate(self.products_df['product_id']):
            product_reviews = self.reviews_df[self.reviews_df['product_id'] == product_id]
            for i, skin_type in enumerate(skin_types_all):
                skin_type_reviews = product_reviews[
                    product_reviews['skin_types'].apply(lambda x: skin_type in x)
                ]
                if len(skin_type_reviews) > 0:
                    skin_ratings[idx, i] = skin_type_reviews['user_rating'].mean() / 5.0
                else:
                    skin_ratings[idx, i] = 1.0

        # ✅ 노드 피처 (TF-IDF SVD만 성분으로 사용)
        node_features = np.concatenate([
            category_onehot,
            brand_onehot,
            self.ingredient_tfidf_svd,
            skin_ratings
        ], axis=1)

        self.node_features = torch.FloatTensor(node_features)

        print("🔗 제품 간 유사도 계산 중...")
        edges = self._compute_product_edges(top_k=15)
        self.edge_index = torch.LongTensor(edges)

        self.data = Data(x=self.node_features, edge_index=self.edge_index)

        print("✅ 그래프 구축 완료")
        print(f"   - 노드 수 (제품): {self.data.x.size(0)}")
        print(f"   - 피처 차원: {self.data.x.size(1)}")
        print(f"   - 엣지 수: {self.data.edge_index.size(1)}")

    def _compute_product_edges(self, top_k=15):
        # normalize 되어 있으니 dot = cosine
        ingredient_sim = self.ingredient_tfidf_svd @ self.ingredient_tfidf_svd.T  # (N,N)

        edges = []
        n = len(self.products_df)

        # 1) 성분 유사도
        for i in range(n):
            sims = ingredient_sim[i].copy()
            sims[i] = -1
            top_indices = np.argsort(sims)[-top_k:]
            for j in top_indices:
                if sims[j] > 0.1:
                    edges.append([i, j])

        # 2) 같은 카테고리
        for category in self.products_df['category'].unique():
            category_products = self.products_df[self.products_df['category'] == category]['product_encoded'].values
            if len(category_products) > 1:
                for a in range(len(category_products)):
                    for b in range(a + 1, min(a + 6, len(category_products))):
                        edges.append([category_products[a], category_products[b]])
                        edges.append([category_products[b], category_products[a]])

        # 3) 협업 필터링
        skin_types_all = ['건성', '민감성', '복합성', '약건성', '지성', '트러블성', '중성']
        for skin_type in skin_types_all:
            skin_reviews = self.reviews_df[
                (self.reviews_df['skin_types'].apply(lambda x: skin_type in x)) &
                (self.reviews_df['user_rating'] >= 4)
            ]
            liked_products = skin_reviews['product_encoded'].unique()

            if len(liked_products) > 1:
                for i, prod_i in enumerate(liked_products):
                    avg_rating_i = skin_reviews[skin_reviews['product_encoded'] == prod_i]['user_rating'].mean()
                    similar_count = 0

                    for prod_j in liked_products[i+1:]:
                        if similar_count >= 10:
                            break
                        avg_rating_j = skin_reviews[skin_reviews['product_encoded'] == prod_j]['user_rating'].mean()
                        if avg_rating_i >= 4.0 and avg_rating_j >= 4.0:
                            edges.append([prod_i, prod_j])
                            edges.append([prod_j, prod_i])
                            similar_count += 1

        edges = list(set(map(tuple, edges)))
        if len(edges) == 0:
            edges = [[i, i+1] for i in range(n-1)] + [[i+1, i] for i in range(n-1)]

        return np.array(edges).T

    # ---------- 학습/평가 ----------
    def train_model(self, epochs=100, hidden_channels=128, lr=0.001):
        print(f"\n🎓 모델 학습 시작 (에폭: {epochs})")

        train_reviews, test_reviews = train_test_split(self.reviews_df, test_size=0.2, random_state=42)

        in_channels = self.node_features.size(1)
        self.model = ProductGNN(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=3).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        self.data = self.data.to(self.device)

        skin_types_all = ['건성', '민감성', '복합성', '약건성', '지성', '트러블성', '중성']
        self.skin_type_to_vector = {st: [1.0 if x == st else 0.0 for x in skin_types_all] for st in skin_types_all}

        best_val_loss = float('inf')

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()

            batch_size = 512
            indices = np.random.choice(len(train_reviews), min(batch_size, len(train_reviews)), replace=False)
            batch = train_reviews.iloc[indices]

            product_indices = torch.LongTensor(batch['product_encoded'].values).to(self.device)
            ratings = torch.FloatTensor(batch['rating_normalized'].values).to(self.device)

            user_features_list = []
            for _, row in batch.iterrows():
                skin_types = row['skin_types']
                skin_vector = [0.0] * 7
                for skin_type in skin_types:
                    if skin_type in self.skin_type_to_vector:
                        tv = self.skin_type_to_vector[skin_type]
                        for i in range(len(tv)):
                            skin_vector[i] += tv[i]

                total = sum(skin_vector)
                if total > 0:
                    skin_vector = [v / total for v in skin_vector]
                else:
                    skin_vector = [1/7] * 7

                user_features_list.append(skin_vector + [row['rating_normalized']])

            user_features = torch.FloatTensor(user_features_list).to(self.device)

            predictions = self.model.predict_rating(self.data.x, self.data.edge_index, user_features, product_indices)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                val_loss = self._validate(test_reviews, criterion)
                metrics = self.evaluate(test_reviews, k=5)

                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")
                print(f"  AUC: {metrics['AUC']:.4f} | Recall@5: {metrics['Recall@5']:.4f} | "
                      f"AP: {metrics['AP']:.4f} | NDCG@5: {metrics['NDCG@5']:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), 'singlenode/best_single_gnn_model.pt')

        print(f"\n✅ 학습 완료! 최고 Validation Loss: {best_val_loss:.4f}")
        self.model.load_state_dict(torch.load('singlenode/best_single_gnn_model.pt'))

    def _validate(self, test_reviews, criterion):
        self.model.eval()
        with torch.no_grad():
            sample_size = min(1000, len(test_reviews))
            test_sample = test_reviews.sample(n=sample_size, random_state=42)

            product_indices = torch.LongTensor(test_sample['product_encoded'].values).to(self.device)
            ratings = torch.FloatTensor(test_sample['rating_normalized'].values).to(self.device)

            user_features_list = []
            for _, row in test_sample.iterrows():
                skin_types = row['skin_types']
                skin_vector = [0.0] * 7
                for skin_type in skin_types:
                    if skin_type in self.skin_type_to_vector:
                        tv = self.skin_type_to_vector[skin_type]
                        for i in range(len(tv)):
                            skin_vector[i] += tv[i]
                s = sum(skin_vector)
                if s > 0:
                    skin_vector = [v / s for v in skin_vector]
                user_features_list.append(skin_vector + [row['rating_normalized']])

            user_features = torch.FloatTensor(user_features_list).to(self.device)
            predictions = self.model.predict_rating(self.data.x, self.data.edge_index, user_features, product_indices)
            loss = criterion(predictions, ratings)
        return loss.item()

    def evaluate(self, test_reviews, k=5):
        self.model.eval()
        with torch.no_grad():
            sample_size = min(3000, len(test_reviews))
            test_sample = test_reviews.sample(n=sample_size, random_state=42)

            product_indices = torch.LongTensor(test_sample['product_encoded'].values).to(self.device)

            user_features_list = []
            for _, row in test_sample.iterrows():
                skin_types = row['skin_types']
                skin_vector = [0.0] * 7
                for skin_type in skin_types:
                    if skin_type in self.skin_type_to_vector:
                        tv = self.skin_type_to_vector[skin_type]
                        for i in range(len(tv)):
                            skin_vector[i] += tv[i]
                s = sum(skin_vector)
                if s > 0:
                    skin_vector = [v / s for v in skin_vector]
                user_features_list.append(skin_vector + [row['rating_normalized']])

            user_features = torch.FloatTensor(user_features_list).to(self.device)

            predictions = self.model.predict_rating(self.data.x, self.data.edge_index, user_features, product_indices)
            predictions = predictions.cpu().numpy()

            true_labels = (test_sample['user_rating'].values >= 4).astype(int)
            predictions = np.clip(predictions, 0, 1)

            try:
                auc = roc_auc_score(true_labels, predictions)
            except:
                auc = 0.5

            try:
                ap = average_precision_score(true_labels, predictions)
            except:
                ap = 0.0

            user_recalls, user_ndcgs = [], []
            for user in test_sample['user_encoded'].unique():
                user_data = test_sample[test_sample['user_encoded'] == user]
                if len(user_data) < 2:
                    continue

                user_pred = predictions[test_sample['user_encoded'] == user]
                user_true = true_labels[test_sample['user_encoded'] == user]

                top_k_indices = np.argsort(user_pred)[-k:]
                top_k_relevant = user_true[top_k_indices]

                if user_true.sum() > 0:
                    recall = top_k_relevant.sum() / min(k, user_true.sum())
                    user_recalls.append(recall)

                try:
                    ndcg = ndcg_score([user_true], [user_pred], k=k)
                    user_ndcgs.append(ndcg)
                except:
                    pass

            recall_at_k = np.mean(user_recalls) if user_recalls else 0.0
            ndcg_at_k = np.mean(user_ndcgs) if user_ndcgs else 0.0

            return {'AUC': auc, 'Recall@5': recall_at_k, 'AP': ap, 'NDCG@5': ndcg_at_k}

    # ---------- 추천 ----------
    def recommend(self, skin_type, category, favorite_product_id=None):
        """
        ✅ category는 반드시 FIXED_CATEGORIES 중 하나로만 받음
        """
        self.validate_category(category)

        top_n = 5
        print(f"\n🔍 추천 생성 중...")
        print(f"   - 피부타입: {skin_type}")
        print(f"   - 카테고리: {category}")
        print(f"   - 선호 제품: {favorite_product_id if favorite_product_id else '없음'}")

        self.model.eval()
        with torch.no_grad():
            skin_vector = self.skin_type_to_vector.get(skin_type, [1/7] * 7)
            user_feature = torch.FloatTensor([skin_vector + [0.8]]).to(self.device)

            all_product_indices = torch.arange(len(self.products_df)).to(self.device)
            scores = self.model.predict_rating(
                self.data.x, self.data.edge_index, user_feature, all_product_indices
            ).cpu().numpy()

            # 선호 제품 유사도 섞기 (성분 TF-IDF SVD)
            if favorite_product_id and favorite_product_id in self.products_df['product_id'].values:
                fav_encoded = self.products_df[self.products_df['product_id'] == favorite_product_id]['product_encoded'].values[0]
                fav_vec = self.ingredient_tfidf_svd[fav_encoded].reshape(1, -1)
                sim = (fav_vec @ self.ingredient_tfidf_svd.T).ravel()
                scores = 0.7 * scores + 0.3 * sim

            # ✅ 카테고리 필터 (완전 일치)
            cat_series = self.products_df['category'].astype(str).str.strip()
            mask = (cat_series == category).values
            scores[~mask] = -np.inf

            # ✅ 전부 -inf면 빈 추천 반환
            if np.isneginf(scores).all():
                print(f"⚠️ '{category}' 카테고리에 해당하는 제품이 없어서 추천 불가")
                return []

            top_indices = np.argsort(scores)[-top_n*2:][::-1]

            recs = []
            for idx in top_indices:
                if len(recs) >= top_n:
                    break
                if np.isneginf(scores[idx]):
                    continue

                product = self.products_df.iloc[idx]
                product_id = product['product_id']

                same_skin_reviews = self.reviews_df[
                    (self.reviews_df['product_id'] == product_id) &
                    (self.reviews_df['skin_types'].apply(lambda x: skin_type in x))
                ].sort_values('user_rating', ascending=False)

                reviews = same_skin_reviews.head(3)['review_text'].tolist()
                if len(reviews) < 3:
                    all_reviews = self.reviews_df[self.reviews_df['product_id'] == product_id].sort_values('user_rating', ascending=False)
                    reviews.extend(all_reviews.head(3 - len(reviews))['review_text'].tolist())

                url = f"https://www.example.com/product/{product_id}"
                main_ingredients = product['ingredients_list'][:5]

                recs.append({
                    'product_name': product['product_name'],
                    'brand': product['brand'],
                    'category': product['category'],
                    'url': url,
                    'main_ingredients': main_ingredients,
                    'reviews': reviews,
                    'rating': product.get('product_rating', np.nan),
                    'predicted_score': float(scores[idx])
                })

            return recs

    def print_recommendations(self, recommendations):
        if not recommendations:
            print("\n(추천 결과 없음)")
            return

        print("\n" + "="*80)
        print("🎁 추천 제품")
        print("="*80)

        for i, rec in enumerate(recommendations, 1):
            print(f"\n【 {i}. {rec['product_name']} 】")
            print(f"   브랜드: {rec['brand']}")
            print(f"   카테고리: {rec['category']}")
            if not np.isnan(rec['rating']):
                print(f"   평점: ⭐ {rec['rating']:.1f}")
            print(f"   예측 점수: {rec['predicted_score']:.3f}")
            print(f"   URL: {rec['url']}")
            print(f"   주요 성분: {', '.join(rec['main_ingredients'])}")
            print(f"\n   💬 같은 피부타입 사용자 리뷰:")
            for j, review in enumerate(rec['reviews'], 1):
                review_short = review[:100] + "..." if len(review) > 100 else review
                print(f"      {j}) {review_short}")
            print("-" * 80)


def main():
    recommender = SingleNodeGNNRecommender(
        products_path='singlenode/final_products.csv',
        reviews_path='singlenode/final_total_reviews.csv',
        svd_dim=100
    )

    recommender.train_model(epochs=50, hidden_channels=128, lr=0.001)

    print("\n" + "="*80)
    print("📊 최종 평가 결과")
    print("="*80)
    test_reviews = recommender.reviews_df.sample(frac=0.2, random_state=42)
    final_metrics = recommender.evaluate(test_reviews, k=5)
    print(f"  AUC: {final_metrics['AUC']:.4f}")
    print(f"  Recall@5: {final_metrics['Recall@5']:.4f}")
    print(f"  AP: {final_metrics['AP']:.4f}")
    print(f"  NDCG@5: {final_metrics['NDCG@5']:.4f}")
    print("="*80)

    print("\n\n" + "="*80)
    print("💡 추천 시스템 테스트(카테고리 5개 고정)")
    print("="*80)

    # ✅ 사용자 입력: 번호로 카테고리 강제
    skin_type = input("피부타입 입력(건성/민감성/복합성/약건성/지성/트러블성/중성): ").strip()
    category = recommender.choose_category_cli()
    favorite = input("선호 제품 ID (없으면 엔터): ").strip()
    favorite = favorite if favorite else None

    recs = recommender.recommend(skin_type=skin_type, category=category, favorite_product_id=favorite)
    recommender.print_recommendations(recs)

    torch.save({
        'model_state': recommender.model.state_dict(),
        'product_encoder': recommender.product_encoder,
        'category_encoder': recommender.category_encoder,
        'brand_encoder': recommender.brand_encoder,
        'mlb_ingredients': recommender.mlb_ingredients,
        'user_encoder': recommender.user_encoder,
        'node_features': recommender.node_features,
        'edge_index': recommender.edge_index,
        'tfidf_vectorizer': recommender.tfidf_vectorizer,
        'svd': recommender.svd,
    }, 'single_node_gnn_recommender.pt')

    print("\n✅ 모델이 'single_node_gnn_recommender.pt'로 저장되었습니다.")


if __name__ == "__main__":
    main()
