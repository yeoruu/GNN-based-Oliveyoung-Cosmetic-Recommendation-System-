"""
단일 노드 GNN 기반 화장품 추천시스템
제품 노드에 피처를 추가하고, 사용자-제품 상호작용을 엣지로 표현
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import roc_auc_score, average_precision_score, ndcg_score
from sklearn.metrics.pairwise import cosine_similarity
import ast
import warnings
warnings.filterwarnings('ignore')


class ProductGNN(nn.Module):
    """단일 노드 (제품) GNN 모델"""
    def __init__(self, in_channels, hidden_channels, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        
        # GNN 레이어들
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # 예측 레이어
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, 1)
        )
        
    def forward(self, x, edge_index):
        """제품 노드 임베딩 생성"""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.3, training=self.training)
        
        return x
    
    def predict_rating(self, x, edge_index, user_features, product_indices):
        """
        사용자-제품 평점 예측
        user_features: 가상 사용자 피처 (피부타입, 선호도 등) [batch_size, feature_dim]
        product_indices: 예측할 제품 인덱스들 [batch_size]
        """
        # 제품 임베딩
        product_emb = self.forward(x, edge_index)
        
        # 사용자 피처와 제품 임베딩 매칭
        # user_features가 이미 배치 형태 [batch_size, feature_dim]
        if user_features.dim() == 1:
            # 단일 사용자인 경우 배치로 확장
            user_emb = user_features.unsqueeze(0).repeat(len(product_indices), 1)
        else:
            # 이미 배치 형태
            user_emb = user_features
        
        selected_product_emb = product_emb[product_indices]
        
        # 결합하여 예측
        combined = torch.cat([user_emb, selected_product_emb], dim=-1)
        return self.predictor(combined).squeeze()


class SingleNodeGNNRecommender:
    """단일 노드 GNN 추천 시스템"""
    
    def __init__(self, products_path, reviews_path):
        print("📚 데이터 로딩 중...")
        self.products_df = pd.read_csv(products_path)
        self.reviews_df = pd.read_csv(reviews_path, encoding='utf-8-sig')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  디바이스: {self.device}")
        
        # 인코더 초기화
        self.product_encoder = LabelEncoder()
        self.category_encoder = LabelEncoder()
        self.brand_encoder = LabelEncoder()
        self.mlb_ingredients = MultiLabelBinarizer()
        self.user_encoder = LabelEncoder()
        
        self._preprocess_data()
        self._build_graph()
        
    def _preprocess_data(self):
        """데이터 전처리"""
        print("🔧 데이터 전처리 중...")
        
        # 제품 데이터 처리
        def parse_ingredients(x):
            if not isinstance(x, str):
                return []
            try:
                # 특수문자 작은따옴표를 일반 작은따옴표로 변환
                x = x.replace(''', "'").replace(''', "'")
                x = x.replace('"', '"').replace('"', '"')
                return ast.literal_eval(x)
            except:
                # 파싱 실패 시 빈 리스트 반환
                return []
        
        self.products_df['ingredients_list'] = self.products_df['ingredients'].apply(parse_ingredients)
        
        # 리뷰 데이터 처리
        # 결측값을 가장 빈도가 높은 피부타입으로 채우기
        if self.reviews_df['user_keywords'].isna().any():
            most_common = self.reviews_df['user_keywords'].mode()
            default_skin = most_common[0] if len(most_common) > 0 else '복합성'
            print(f"⚠️  피부타입 결측값을 '{default_skin}'로 채웁니다.")
            self.reviews_df['user_keywords'] = self.reviews_df['user_keywords'].fillna(default_skin)
        
        self.reviews_df['skin_types'] = self.reviews_df['user_keywords'].apply(
            lambda x: [t.strip() for t in x.split('|')]
        )
        
        # 평점을 0-1로 정규화
        self.reviews_df['rating_normalized'] = self.reviews_df['user_rating'] / 5.0
        
        # 인코딩
        self.products_df['product_encoded'] = self.product_encoder.fit_transform(
            self.products_df['product_id']
        )
        self.products_df['category_encoded'] = self.category_encoder.fit_transform(
            self.products_df['category']
        )
        self.products_df['brand_encoded'] = self.brand_encoder.fit_transform(
            self.products_df['brand']
        )
        self.reviews_df['user_encoded'] = self.user_encoder.fit_transform(
            self.reviews_df['user_id']
        )
        
        # 리뷰에 제품 인코딩 매핑
        product_to_encoded = dict(zip(
            self.products_df['product_id'], 
            self.products_df['product_encoded']
        ))
        self.reviews_df['product_encoded'] = self.reviews_df['product_id'].map(
            product_to_encoded
        )
        
        # 성분 one-hot 인코딩
        all_ingredients = self.products_df['ingredients_list'].tolist()
        self.ingredient_matrix = self.mlb_ingredients.fit_transform(all_ingredients)
        
        print(f"✅ 제품 수: {len(self.product_encoder.classes_)}")
        print(f"✅ 사용자 수: {len(self.user_encoder.classes_)}")
        print(f"✅ 리뷰 수: {len(self.reviews_df)}")
        print(f"✅ 카테고리 수: {len(self.category_encoder.classes_)}")
        print(f"✅ 브랜드 수: {len(self.brand_encoder.classes_)}")
        print(f"✅ 성분 수: {self.ingredient_matrix.shape[1]}")
        
    def _build_graph(self):
        """단일 노드 (제품) 그래프 구축"""
        print("🔨 그래프 구축 중...")
        
        # 제품 노드 피처 구성
        # 1. 카테고리 원-핫 인코딩
        num_categories = len(self.category_encoder.classes_)
        category_onehot = np.eye(num_categories)[self.products_df['category_encoded']]
        
        # 2. 브랜드 원-핫 인코딩
        num_brands = len(self.brand_encoder.classes_)
        brand_onehot = np.eye(num_brands)[self.products_df['brand_encoded']]
        
        # 3. 성분 정보
        ingredient_features = self.ingredient_matrix
        
        # 4. 피부타입별 평균 평점 (각 제품이 어떤 피부타입에게 얼마나 좋은 평가를 받았는지)
        skin_types_all = ['건성', '민감성', '복합성', '약건성', '지성', '트러블성', '중성']
        skin_ratings = np.zeros((len(self.products_df), len(skin_types_all)))
        
        for idx, product_id in enumerate(self.products_df['product_id']):
            product_reviews = self.reviews_df[self.reviews_df['product_id'] == product_id]
            for i, skin_type in enumerate(skin_types_all):
                # 해당 피부타입 사용자들의 평점 평균
                skin_type_reviews = product_reviews[
                    product_reviews['skin_types'].apply(lambda x: skin_type in x)
                ]
                if len(skin_type_reviews) > 0:
                    avg_rating = skin_type_reviews['user_rating'].mean() / 5.0
                    skin_ratings[idx, i] = avg_rating
                else:
                    # 해당 피부타입의 리뷰가 없으면 0.0 (인기 없음을 의미)
                    skin_ratings[idx, i] = 0.0
        
        # 모든 피처 결합
        node_features = np.concatenate([
            category_onehot,        # 카테고리
            brand_onehot,           # 브랜드
            ingredient_features,    # 성분
            skin_ratings            # 피부타입별 평균 평점
        ], axis=1)
        
        self.node_features = torch.FloatTensor(node_features)
        
        # 엣지 구축 (제품 간 유사도)
        print("🔗 제품 간 유사도 계산 중...")
        edges = self._compute_product_edges(top_k=15)
        self.edge_index = torch.LongTensor(edges)
        
        # PyG Data 객체 생성
        self.data = Data(
            x=self.node_features,
            edge_index=self.edge_index
        )
        
        print(f"✅ 그래프 구축 완료")
        print(f"   - 노드 수 (제품): {self.data.x.size(0)}")
        print(f"   - 피처 차원: {self.data.x.size(1)}")
        print(f"   - 엣지 수: {self.data.edge_index.size(1)}")
        
    def _compute_product_edges(self, top_k=15):
        """
        제품 간 엣지 생성
        - 성분 유사도
        - 같은 카테고리
        - 협업 필터링 (같은 피부타입 사용자들이 좋아한 제품)
        """
        # 1. 성분 유사도
        ingredient_sim = cosine_similarity(self.ingredient_matrix)
        
        edges = []
        
        # 성분 유사도 기반 엣지
        for i in range(len(self.products_df)):
            similarities = ingredient_sim[i].copy()
            similarities[i] = -1  # 자기 자신 제외
            
            top_indices = np.argsort(similarities)[-top_k:]
            for j in top_indices:
                if similarities[j] > 0.1:
                    edges.append([i, j])
        
        # 2. 같은 카테고리 제품 연결
        for category in self.products_df['category'].unique():
            category_products = self.products_df[
                self.products_df['category'] == category
            ]['product_encoded'].values
            
            if len(category_products) > 1:
                for i in range(len(category_products)):
                    for j in range(i+1, min(i+6, len(category_products))):
                        edges.append([category_products[i], category_products[j]])
                        edges.append([category_products[j], category_products[i]])
        
        # 3. 협업 필터링: 같은 피부타입 사용자들이 높은 평점을 준 제품들 연결
        skin_types_all = ['건성', '민감성', '복합성', '약건성', '지성', '트러블성', '중성']
        
        for skin_type in skin_types_all:
            # 특정 피부타입을 가진 사용자들의 리뷰
            skin_reviews = self.reviews_df[
                (self.reviews_df['skin_types'].apply(lambda x: skin_type in x)) &
                (self.reviews_df['user_rating'] >= 4)
            ]
            
            # 해당 피부타입이 좋아하는 제품들
            liked_products = skin_reviews['product_encoded'].unique()
            
            # 같은 피부타입이 좋아하는 제품들끼리 연결
            if len(liked_products) > 1:
                # 너무 많은 엣지 방지: 각 제품당 최대 10개 연결
                for i, prod_i in enumerate(liked_products):
                    # 해당 제품의 평점들
                    prod_i_ratings = skin_reviews[
                        skin_reviews['product_encoded'] == prod_i
                    ]['user_rating'].values
                    avg_rating_i = prod_i_ratings.mean()
                    
                    # 유사하게 좋은 평가를 받은 제품들과 연결
                    similar_count = 0
                    for prod_j in liked_products[i+1:]:
                        if similar_count >= 10:  # 최대 10개
                            break
                        
                        prod_j_ratings = skin_reviews[
                            skin_reviews['product_encoded'] == prod_j
                        ]['user_rating'].values
                        avg_rating_j = prod_j_ratings.mean()
                        
                        # 둘 다 평점이 높으면 연결
                        if avg_rating_i >= 4.0 and avg_rating_j >= 4.0:
                            edges.append([prod_i, prod_j])
                            edges.append([prod_j, prod_i])
                            similar_count += 1
        
        # 중복 제거
        edges = list(set(map(tuple, edges)))
        
        if len(edges) == 0:
            # 엣지가 없으면 기본 엣지 생성 (순차적 연결)
            edges = [[i, i+1] for i in range(len(self.products_df)-1)]
            edges += [[i+1, i] for i in range(len(self.products_df)-1)]
        
        return np.array(edges).T
    
    def train_model(self, epochs=100, hidden_channels=128, lr=0.001):
        """모델 학습"""
        print(f"\n🎓 모델 학습 시작 (에폭: {epochs})")
        
        # Train/Test 분할
        train_reviews, test_reviews = train_test_split(
            self.reviews_df, test_size=0.2, random_state=42
        )
        
        # 모델 초기화
        in_channels = self.node_features.size(1)
        self.model = ProductGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=3
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()
        
        # 데이터를 디바이스로 이동
        self.data = self.data.to(self.device)
        
        # 사용자 피처 캐싱 (피부타입 원-핫)
        skin_types_all = ['건성', '민감성', '복합성', '약건성', '지성', '트러블성', '중성']
        self.skin_type_to_vector = {}
        for skin_type in skin_types_all:
            vector = [1.0 if st == skin_type else 0.0 for st in skin_types_all]
            self.skin_type_to_vector[skin_type] = vector
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            
            # 배치 샘플링
            batch_size = 512
            indices = np.random.choice(len(train_reviews), min(batch_size, len(train_reviews)), replace=False)
            batch = train_reviews.iloc[indices]
            
            # 배치 데이터 준비
            product_indices = torch.LongTensor(batch['product_encoded'].values).to(self.device)
            ratings = torch.FloatTensor(batch['rating_normalized'].values).to(self.device)
            
            # 사용자 피처 생성 (피부타입 기반)
            user_features_list = []
            for _, row in batch.iterrows():
                skin_types = row['skin_types']
                # 여러 피부타입의 평균
                skin_vector = [0.0] * 4
                for skin_type in skin_types:
                    if skin_type in self.skin_type_to_vector:
                        type_vector = self.skin_type_to_vector[skin_type]
                        for i in range(len(type_vector)):
                            skin_vector[i] += type_vector[i]
                
                # 정규화
                total = sum(skin_vector)
                if total > 0:
                    skin_vector = [v / total for v in skin_vector]
                else:
                    # 피부타입 정보가 없으면 균등 분포
                    skin_vector = [0.25, 0.25, 0.25, 0.25]
                
                user_features_list.append(skin_vector + [row['rating_normalized']])
            
            user_features = torch.FloatTensor(user_features_list).to(self.device)
            
            # Forward
            predictions = self.model.predict_rating(
                self.data.x,
                self.data.edge_index,
                user_features,
                product_indices
            )
            
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            
            # 검증
            if (epoch + 1) % 10 == 0:
                val_loss = self._validate(test_reviews, criterion)
                metrics = self.evaluate(test_reviews, k=5)
                
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")
                print(f"  AUC: {metrics['AUC']:.4f} | Recall@5: {metrics['Recall@5']:.4f} | "
                      f"AP: {metrics['AP']:.4f} | NDCG@5: {metrics['NDCG@5']:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), '/home/claude/best_single_gnn_model.pt')
        
        print(f"\n✅ 학습 완료! 최고 Validation Loss: {best_val_loss:.4f}")
        self.model.load_state_dict(torch.load('/home/claude/best_single_gnn_model.pt'))
    
    def _validate(self, test_reviews, criterion):
        """검증"""
        self.model.eval()
        
        with torch.no_grad():
            # 샘플링
            sample_size = min(1000, len(test_reviews))
            test_sample = test_reviews.sample(n=sample_size, random_state=42)
            
            product_indices = torch.LongTensor(test_sample['product_encoded'].values).to(self.device)
            ratings = torch.FloatTensor(test_sample['rating_normalized'].values).to(self.device)
            
            # 사용자 피처
            user_features_list = []
            for _, row in test_sample.iterrows():
                skin_types = row['skin_types']
                skin_vector = [0.0] * 4
                for skin_type in skin_types:
                    if skin_type in self.skin_type_to_vector:
                        for i, val in enumerate(self.skin_type_to_vector[skin_type]):
                            skin_vector[i] += val
                
                if sum(skin_vector) > 0:
                    skin_vector = [v / sum(skin_vector) for v in skin_vector]
                
                user_features_list.append(skin_vector + [row['rating_normalized']])
            
            user_features = torch.FloatTensor(user_features_list).to(self.device)
            
            predictions = self.model.predict_rating(
                self.data.x,
                self.data.edge_index,
                user_features,
                product_indices
            )
            
            loss = criterion(predictions, ratings)
        
        return loss.item()
    
    def evaluate(self, test_reviews, k=5):
        """평가 지표 계산"""
        self.model.eval()
        
        with torch.no_grad():
            # 샘플링
            sample_size = min(3000, len(test_reviews))
            test_sample = test_reviews.sample(n=sample_size, random_state=42)
            
            product_indices = torch.LongTensor(test_sample['product_encoded'].values).to(self.device)
            
            # 사용자 피처
            user_features_list = []
            for _, row in test_sample.iterrows():
                skin_types = row['skin_types']
                skin_vector = [0.0] * 4
                for skin_type in skin_types:
                    if skin_type in self.skin_type_to_vector:
                        for i, val in enumerate(self.skin_type_to_vector[skin_type]):
                            skin_vector[i] += val
                
                if sum(skin_vector) > 0:
                    skin_vector = [v / sum(skin_vector) for v in skin_vector]
                
                user_features_list.append(skin_vector + [row['rating_normalized']])
            
            user_features = torch.FloatTensor(user_features_list).to(self.device)
            
            predictions = self.model.predict_rating(
                self.data.x,
                self.data.edge_index,
                user_features,
                product_indices
            )
            
            predictions = predictions.cpu().numpy()
            true_labels = (test_sample['user_rating'].values >= 4).astype(int)
            
            # 예측값을 0-1 범위로 클리핑
            predictions = np.clip(predictions, 0, 1)
            
            # AUC
            try:
                auc = roc_auc_score(true_labels, predictions)
            except:
                auc = 0.5
            
            # AP
            try:
                ap = average_precision_score(true_labels, predictions)
            except:
                ap = 0.0
            
            # Recall@K 및 NDCG@K
            user_recalls = []
            user_ndcgs = []
            
            for user in test_sample['user_encoded'].unique():
                user_data = test_sample[test_sample['user_encoded'] == user]
                if len(user_data) < 2:
                    continue
                
                user_pred = predictions[test_sample['user_encoded'] == user]
                user_true = true_labels[test_sample['user_encoded'] == user]
                
                # Top-K
                top_k_indices = np.argsort(user_pred)[-k:]
                top_k_relevant = user_true[top_k_indices]
                
                # Recall@K
                if user_true.sum() > 0:
                    recall = top_k_relevant.sum() / min(k, user_true.sum())
                    user_recalls.append(recall)
                
                # NDCG@K
                try:
                    ndcg = ndcg_score([user_true], [user_pred], k=k)
                    user_ndcgs.append(ndcg)
                except:
                    pass
            
            recall_at_k = np.mean(user_recalls) if user_recalls else 0.0
            ndcg_at_k = np.mean(user_ndcgs) if user_ndcgs else 0.0
            
            return {
                'AUC': auc,
                'Recall@5': recall_at_k,
                'AP': ap,
                'NDCG@5': ndcg_at_k
            }
    
    def recommend(self, skin_type, category=None, favorite_product_id=None):
        """
        제품 추천 (5개 고정)
        
        Args:
            skin_type: 피부 타입 (건성, 민감성, 복합성, 약건성, 지성, 트러블성, 중성)
            category: 카테고리 (선택사항, None이면 전체)
            favorite_product_id: 좋아하는 제품 ID (선택사항)
        
        Returns:
            추천 제품 리스트 (5개)
        """
        top_n = 5  # 고정
        
        print(f"\n🔍 추천 생성 중...")
        print(f"   - 피부타입: {skin_type}")
        print(f"   - 카테고리: {category if category else '전체'}")
        print(f"   - 선호 제품: {favorite_product_id if favorite_product_id else '없음'}")
        
        self.model.eval()
        
        with torch.no_grad():
            # 가상 사용자 피처 생성
            skin_vector = self.skin_type_to_vector.get(skin_type, [1/7] * 7)
            user_feature = torch.FloatTensor([skin_vector + [0.8]]).to(self.device)
            
            # 선호 제품이 있으면 반영
            if favorite_product_id and favorite_product_id in self.products_df['product_id'].values:
                fav_encoded = self.products_df[
                    self.products_df['product_id'] == favorite_product_id
                ]['product_encoded'].values[0]
                
                # 선호 제품의 평점을 높게 설정
                fav_rating = torch.FloatTensor([[1.0]]).to(self.device)
            
            # 전체 제품에 대한 예측
            all_product_indices = torch.arange(len(self.products_df)).to(self.device)
            
            scores = self.model.predict_rating(
                self.data.x,
                self.data.edge_index,
                user_feature,
                all_product_indices
            ).cpu().numpy()
            
            # 선호 제품이 있으면 유사 제품 가중치 부여
            if favorite_product_id and favorite_product_id in self.products_df['product_id'].values:
                fav_encoded = self.products_df[
                    self.products_df['product_id'] == favorite_product_id
                ]['product_encoded'].values[0]
                
                # 성분 유사도 계산
                fav_ingredients = self.ingredient_matrix[fav_encoded]
                ingredient_sim = cosine_similarity([fav_ingredients], self.ingredient_matrix)[0]
                
                # 유사도 가중치 적용
                scores = 0.7 * scores + 0.3 * ingredient_sim
            
            # 카테고리 필터링
            if category and category != '전체':
                category_mask = self.products_df['category'] == category
                scores[~category_mask.values] = -np.inf
            
            # Top-N 선택
            top_indices = np.argsort(scores)[-top_n*2:][::-1]
            
            # 추천 리스트 생성
            recommendations = []
            for idx in top_indices:
                if len(recommendations) >= top_n:
                    break
                
                product = self.products_df.iloc[idx]
                product_id = product['product_id']
                
                # 같은 스킨타입 사용자 리뷰
                same_skin_reviews = self.reviews_df[
                    (self.reviews_df['product_id'] == product_id) &
                    (self.reviews_df['skin_types'].apply(lambda x: skin_type in x))
                ].sort_values('user_rating', ascending=False)
                
                reviews = same_skin_reviews.head(3)['review_text'].tolist()
                
                # 리뷰가 부족하면 전체 리뷰에서 가져오기
                if len(reviews) < 3:
                    all_reviews = self.reviews_df[
                        self.reviews_df['product_id'] == product_id
                    ].sort_values('user_rating', ascending=False)
                    additional = all_reviews.head(3 - len(reviews))['review_text'].tolist()
                    reviews.extend(additional)
                
                # URL 생성
                url = f"https://www.example.com/product/{product_id}"
                
                # 주요 성분 (상위 5개)
                main_ingredients = product['ingredients_list'][:5]
                
                recommendations.append({
                    'product_name': product['product_name'],
                    'brand': product['brand'],
                    'url': url,
                    'main_ingredients': main_ingredients,
                    'reviews': reviews,
                    'rating': product['product_rating'],
                    'predicted_score': scores[idx]
                })
            
            return recommendations
    
    def print_recommendations(self, recommendations):
        """추천 결과 출력"""
        print("\n" + "="*80)
        print("🎁 추천 제품")
        print("="*80)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n【 {i}. {rec['product_name']} 】")
            print(f"   브랜드: {rec['brand']}")
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
    """메인 실행 함수"""
    
    # 시스템 초기화
    recommender = SingleNodeGNNRecommender(
        products_path='final_products.csv',
        reviews_path='final_total_reviews.csv'
    )
    
    # 모델 학습
    recommender.train_model(epochs=50, hidden_channels=128, lr=0.001)
    
    # 최종 평가
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
    
    # 추천 예시
    print("\n\n" + "="*80)
    print("💡 추천 시스템 테스트")
    print("="*80)
    
    # 예시 1: 건성 피부, 로션 카테고리
    recommendations = recommender.recommend(
        skin_type='건성',
        category='로션',
        favorite_product_id='L1'
    )
    recommender.print_recommendations(recommendations)
    
    # 예시 2: 지성 피부, 전체 카테고리
    print("\n\n")
    recommendations = recommender.recommend(
        skin_type='지성',
        category=None,
        favorite_product_id=None
    )
    recommender.print_recommendations(recommendations)
    
    # 예시 3: 민감성 피부, 세럼 카테고리
    print("\n\n")
    recommendations = recommender.recommend(
        skin_type='민감성',
        category='세럼',
        favorite_product_id=None
    )
    recommender.print_recommendations(recommendations)
    
    # 모델 저장
    torch.save({
        'model_state': recommender.model.state_dict(),
        'product_encoder': recommender.product_encoder,
        'category_encoder': recommender.category_encoder,
        'brand_encoder': recommender.brand_encoder,
        'mlb_ingredients': recommender.mlb_ingredients,
        'user_encoder': recommender.user_encoder,
        'node_features': recommender.node_features,
        'edge_index': recommender.edge_index,
    }, 'single_node_gnn_recommender.pt')
    
    print("\n✅ 모델이 'single_node_gnn_recommender.pt'로 저장되었습니다.")


if __name__ == "__main__":
    main()
