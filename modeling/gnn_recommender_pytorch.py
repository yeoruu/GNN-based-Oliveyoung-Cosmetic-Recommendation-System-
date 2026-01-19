"""
PyTorch Geometric 기반 GNN 화장품 추천 시스템
실제 데이터로 실행 가능한 완전한 코드

필수 라이브러리 설치:
pip install torch torch-geometric scikit-learn pandas numpy

실행 방법:
python gnn_recommender_pytorch.py
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error,
    accuracy_score, 
    precision_recall_fscore_support,
    ndcg_score,
    r2_score
)
from collections import defaultdict, Counter
import ast
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================================
# 설정
# ============================================================================
class Config:
    """하이퍼파라미터 설정"""
    # 파일 경로
    PRODUCTS_FILE = 'final_products.csv'
    REVIEWS_FILE = 'final_total_reviews.csv'
    
    # 모델 하이퍼파라미터
    FEATURE_DIM = 64
    HIDDEN_DIM = 128
    EMBEDDING_DIM = 64
    NUM_GNN_LAYERS = 3
    DROPOUT = 0.5
    
    # 학습 설정
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 5e-4
    NUM_EPOCHS = 150
    PATIENCE = 20
    
    # 기타
    TEST_SIZE = 0.2
    RANDOM_SEED = 42
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

print("=" * 80)
print("PyTorch GNN 기반 화장품 추천 시스템".center(80))
print("=" * 80)
print(f"\n⚙️  설정:")
print(f"   Device: {config.DEVICE}")
print(f"   Feature Dim: {config.FEATURE_DIM}")
print(f"   Hidden Dim: {config.HIDDEN_DIM}")
print(f"   Learning Rate: {config.LEARNING_RATE}")
print(f"   Epochs: {config.NUM_EPOCHS}")


# ============================================================================
# 1. 데이터 로드 및 전처리
# ============================================================================
class DataLoader:
    """데이터 로더 클래스"""
    
    def __init__(self, products_path, reviews_path):
        self.products_path = products_path
        self.reviews_path = reviews_path
        self.products_df = None
        self.reviews_df = None
        
        # 매핑
        self.ingredient_to_idx = {}
        self.product_to_idx = {}
        self.user_to_idx = {}
        self.idx_to_ingredient = {}
        self.idx_to_product = {}
        self.idx_to_user = {}
        
    def load(self):
        """데이터 로드"""
        print("\n📂 [1] 데이터 로드 중...")
        
        self.products_df = pd.read_csv(self.products_path)
        self.reviews_df = pd.read_csv(self.reviews_path)
        
        print(f"   ✓ 제품: {len(self.products_df):,}개")
        print(f"   ✓ 리뷰: {len(self.reviews_df):,}개")
        
        return self
    
    def parse_ingredients(self, ing_str):
        """성분 파싱"""
        if pd.isna(ing_str):
            return []
        try:
            if isinstance(ing_str, str):
                ing_list = ast.literal_eval(ing_str)
                return [str(ing).strip() for ing in ing_list if ing]
            return []
        except:
            return []
    
    def preprocess(self):
        """전처리"""
        print("\n🔧 [2] 데이터 전처리 중...")
        
        # 성분 파싱
        self.products_df['ingredient_list'] = self.products_df['ingredients'].apply(
            self.parse_ingredients
        )
        
        # 유효한 제품만
        self.products_df = self.products_df[
            self.products_df['ingredient_list'].apply(len) > 0
        ]
        
        # 고유 성분
        all_ingredients = set()
        for ings in self.products_df['ingredient_list']:
            all_ingredients.update(ings)
        
        print(f"   ✓ 유효한 제품: {len(self.products_df):,}개")
        print(f"   ✓ 고유 성분: {len(all_ingredients):,}개")
        
        # 인덱스 매핑
        self.ingredient_to_idx = {ing: idx for idx, ing in enumerate(sorted(all_ingredients))}
        self.product_to_idx = {pid: idx for idx, pid in enumerate(self.products_df['product_id'].unique())}
        self.user_to_idx = {uid: idx for idx, uid in enumerate(self.reviews_df['user_id'].unique())}
        
        # 역매핑
        self.idx_to_ingredient = {v: k for k, v in self.ingredient_to_idx.items()}
        self.idx_to_product = {v: k for k, v in self.product_to_idx.items()}
        self.idx_to_user = {v: k for k, v in self.user_to_idx.items()}
        
        # 인덱스 추가
        self.products_df['product_idx'] = self.products_df['product_id'].map(self.product_to_idx)
        self.reviews_df['user_idx'] = self.reviews_df['user_id'].map(self.user_to_idx)
        self.reviews_df['product_idx'] = self.reviews_df['product_id'].map(self.product_to_idx)
        
        # NaN 제거
        self.reviews_df = self.reviews_df.dropna(subset=['user_idx', 'product_idx', 'user_rating'])
        
        print(f"   ✓ 사용자: {len(self.user_to_idx):,}명")
        print(f"   ✓ 처리된 리뷰: {len(self.reviews_df):,}개")
        
        return self


# ============================================================================
# 2. 그래프 생성
# ============================================================================
class GraphBuilder:
    """그래프 빌더"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.num_products = len(data_loader.product_to_idx)
        self.num_ingredients = len(data_loader.ingredient_to_idx)
        self.num_users = len(data_loader.user_to_idx)
        self.num_nodes = self.num_products + self.num_ingredients + self.num_users
        
        self.edge_index = None
        self.node_features = None
        
    def build_edges(self):
        """엣지 생성"""
        print("\n🕸️  [3] 그래프 엣지 생성 중...")
        
        edges = []
        
        # Product ↔ Ingredient
        for _, row in self.data_loader.products_df.iterrows():
            prod_idx = row['product_idx']
            for ing in row['ingredient_list']:
                if ing in self.data_loader.ingredient_to_idx:
                    ing_idx = self.data_loader.ingredient_to_idx[ing] + self.num_products
                    edges.append([prod_idx, ing_idx])
                    edges.append([ing_idx, prod_idx])
        
        prod_ing_edges = len(edges)
        
        # User ↔ Product
        for _, row in self.data_loader.reviews_df.iterrows():
            user_idx = int(row['user_idx']) + self.num_products + self.num_ingredients
            prod_idx = int(row['product_idx'])
            edges.append([user_idx, prod_idx])
            edges.append([prod_idx, user_idx])
        
        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        print(f"   ✓ Product-Ingredient 엣지: {prod_ing_edges:,}개")
        print(f"   ✓ User-Product 엣지: {len(edges) - prod_ing_edges:,}개")
        print(f"   ✓ 총 엣지: {len(edges):,}개")
        
        return self
    
    def build_features(self):
        """노드 특징 생성"""
        print("\n🎨 [4] 노드 특징 생성 중...")
        
        # 랜덤 초기화
        self.node_features = torch.randn(self.num_nodes, config.FEATURE_DIM)
        
        # 제품 특징 강화 (카테고리 기반)
        categories = self.data_loader.products_df['category'].unique()
        category_to_idx = {cat: i for i, cat in enumerate(categories)}
        
        for _, row in self.data_loader.products_df.iterrows():
            idx = row['product_idx']
            if idx < self.num_products:
                cat_idx = category_to_idx.get(row['category'], 0)
                # 카테고리 정보 인코딩
                if cat_idx < config.FEATURE_DIM:
                    self.node_features[idx, cat_idx] = 1.0
        
        print(f"   ✓ 노드 특징: {self.node_features.shape}")
        
        return self
    
    def get_data(self):
        """PyG Data 객체 반환"""
        return Data(
            x=self.node_features.to(config.DEVICE),
            edge_index=self.edge_index.to(config.DEVICE)
        )


# ============================================================================
# 3. GNN 모델
# ============================================================================
class IngredientGNN(nn.Module):
    """성분 기반 GNN"""
    
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_layers=3, dropout=0.5):
        super(IngredientGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GCN 레이어
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        # 배치 정규화
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
    
    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


class RecommendationModel(nn.Module):
    """추천 모델"""
    
    def __init__(self, num_products, num_ingredients, num_users):
        super(RecommendationModel, self).__init__()
        
        self.num_products = num_products
        self.num_ingredients = num_ingredients
        self.num_users = num_users
        
        # GNN
        self.gnn = IngredientGNN(
            in_channels=config.FEATURE_DIM,
            hidden_channels=config.HIDDEN_DIM,
            out_channels=config.EMBEDDING_DIM,
            num_layers=config.NUM_GNN_LAYERS,
            dropout=config.DROPOUT
        )
        
        # 예측 MLP
        self.predictor = nn.Sequential(
            nn.Linear(config.EMBEDDING_DIM * 2, config.HIDDEN_DIM),
            nn.BatchNorm1d(config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.HIDDEN_DIM, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, edge_index, user_indices, product_indices):
        # GNN 임베딩
        embeddings = self.gnn(x, edge_index)
        
        # 유저/제품 임베딩
        user_offset = self.num_products + self.num_ingredients
        user_emb = embeddings[user_indices + user_offset]
        product_emb = embeddings[product_indices]
        
        # 결합
        combined = torch.cat([user_emb, product_emb], dim=1)
        
        # 예측 (1~5 범위)
        rating = self.predictor(combined).squeeze()
        rating = torch.sigmoid(rating) * 4 + 1
        
        return rating


# ============================================================================
# 4. 학습
# ============================================================================
class Trainer:
    """모델 학습"""
    
    def __init__(self, model, graph_data, data_loader):
        self.model = model
        self.graph_data = graph_data
        self.data_loader = data_loader
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.criterion = nn.MSELoss()
        
        # 데이터 분할
        self.train_reviews, self.test_reviews = train_test_split(
            data_loader.reviews_df,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_SEED
        )
        
        print(f"\n📊 [5] 데이터 분할:")
        print(f"   Train: {len(self.train_reviews):,}개")
        print(f"   Test: {len(self.test_reviews):,}개")
        
        self._prepare_tensors()
    
    def _prepare_tensors(self):
        """텐서 준비"""
        # Train
        self.train_user_idx = torch.tensor(
            self.train_reviews['user_idx'].values, dtype=torch.long
        ).to(config.DEVICE)
        self.train_product_idx = torch.tensor(
            self.train_reviews['product_idx'].values, dtype=torch.long
        ).to(config.DEVICE)
        self.train_ratings = torch.tensor(
            self.train_reviews['user_rating'].values, dtype=torch.float
        ).to(config.DEVICE)
        
        # Test
        self.test_user_idx = torch.tensor(
            self.test_reviews['user_idx'].values, dtype=torch.long
        ).to(config.DEVICE)
        self.test_product_idx = torch.tensor(
            self.test_reviews['product_idx'].values, dtype=torch.long
        ).to(config.DEVICE)
        self.test_ratings = torch.tensor(
            self.test_reviews['user_rating'].values, dtype=torch.float
        ).to(config.DEVICE)
    
    def train(self):
        """학습 실행"""
        print(f"\n🚀 [6] 모델 학습 시작 (Epochs: {config.NUM_EPOCHS})...")
        print("=" * 80)
        
        best_test_rmse = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'test_rmse': []}
        
        for epoch in range(config.NUM_EPOCHS):
            # 학습
            self.model.train()
            self.optimizer.zero_grad()
            
            predictions = self.model(
                self.graph_data.x, 
                self.graph_data.edge_index,
                self.train_user_idx, 
                self.train_product_idx
            )
            
            loss = self.criterion(predictions, self.train_ratings)
            loss.backward()
            self.optimizer.step()
            
            history['train_loss'].append(loss.item())
            
            # 평가
            if (epoch + 1) % 10 == 0:
                test_rmse = self._evaluate()
                history['test_rmse'].append(test_rmse)
                
                print(f"Epoch {epoch+1:3d}/{config.NUM_EPOCHS} | "
                      f"Train Loss: {loss.item():.4f} | "
                      f"Test RMSE: {test_rmse:.4f}")
                
                # Early stopping
                if test_rmse < best_test_rmse:
                    best_test_rmse = test_rmse
                    patience_counter = 0
                    torch.save(self.model.state_dict(), 'best_gnn_model.pt')
                else:
                    patience_counter += 1
                
                if patience_counter >= config.PATIENCE // 10:
                    print(f"\n⚠️  Early stopping at epoch {epoch+1}")
                    break
        
        print(f"\n✅ 학습 완료! Best RMSE: {best_test_rmse:.4f}")
        
        return history
    
    def _evaluate(self):
        """평가"""
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(
                self.graph_data.x,
                self.graph_data.edge_index,
                self.test_user_idx,
                self.test_product_idx
            )
            rmse = torch.sqrt(self.criterion(predictions, self.test_ratings))
        
        return rmse.item()


# ============================================================================
# 5. 평가
# ============================================================================
class Evaluator:
    """모델 평가"""
    
    def __init__(self, model, graph_data, test_reviews, data_loader):
        self.model = model
        self.graph_data = graph_data
        self.test_reviews = test_reviews
        self.data_loader = data_loader
    
    def evaluate(self):
        """종합 평가"""
        print("\n" + "=" * 80)
        print("📈 [7] 모델 평가".center(80))
        print("=" * 80)
        
        # 최고 모델 로드
        self.model.load_state_dict(torch.load('best_gnn_model.pt'))
        self.model.eval()
        
        # 텐서 준비
        test_user_idx = torch.tensor(
            self.test_reviews['user_idx'].values, dtype=torch.long
        ).to(config.DEVICE)
        test_product_idx = torch.tensor(
            self.test_reviews['product_idx'].values, dtype=torch.long
        ).to(config.DEVICE)
        test_ratings = torch.tensor(
            self.test_reviews['user_rating'].values, dtype=torch.float
        ).to(config.DEVICE)
        
        # 예측
        with torch.no_grad():
            predictions = self.model(
                self.graph_data.x,
                self.graph_data.edge_index,
                test_user_idx,
                test_product_idx
            )
        
        y_true = test_ratings.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        
        # 평가 지표
        metrics = {}
        
        # 1. Regression
        print("\n📊 Regression Metrics:")
        metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['MAE'] = mean_absolute_error(y_true, y_pred)
        metrics['R2'] = r2_score(y_true, y_pred)
        
        print(f"   • RMSE: {metrics['RMSE']:.4f}")
        print(f"   • MAE: {metrics['MAE']:.4f}")
        print(f"   • R²: {metrics['R2']:.4f}")
        
        # 2. Classification
        print("\n📊 Classification Metrics:")
        y_true_class = np.round(y_true).astype(int)
        y_pred_class = np.round(y_pred).astype(int)
        
        metrics['Accuracy'] = accuracy_score(y_true_class, y_pred_class)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_class, y_pred_class, average='weighted', zero_division=0
        )
        
        print(f"   • Accuracy: {metrics['Accuracy']:.4f}")
        print(f"   • Precision: {precision:.4f}")
        print(f"   • Recall: {recall:.4f}")
        print(f"   • F1-Score: {f1:.4f}")
        
        # 3. Ranking
        print("\n📊 Ranking Metrics:")
        metrics['HR@5'] = self._hit_rate(y_true, y_pred, k=5)
        metrics['HR@10'] = self._hit_rate(y_true, y_pred, k=10)
        
        print(f"   • Hit Rate @5: {metrics['HR@5']:.4f}")
        print(f"   • Hit Rate @10: {metrics['HR@10']:.4f}")
        
        return metrics
    
    def _hit_rate(self, y_true, y_pred, k=10, threshold=4.0):
        """Hit Rate 계산"""
        user_groups = defaultdict(list)
        
        for user_idx, rating, pred in zip(
            self.test_reviews['user_idx'].values, y_true, y_pred
        ):
            user_groups[user_idx].append((rating, pred))
        
        hits = 0
        total = 0
        
        for user_idx, items in user_groups.items():
            if len(items) < k:
                continue
            
            items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
            top_k = items_sorted[:k]
            
            if any(rating >= threshold for rating, _ in top_k):
                hits += 1
            total += 1
        
        return hits / total if total > 0 else 0


# ============================================================================
# 6. 추천 시스템
# ============================================================================
class RecommendationSystem:
    """추천 시스템"""
    
    def __init__(self, model, graph_data, data_loader):
        self.model = model
        self.graph_data = graph_data
        self.data_loader = data_loader
    
    def recommend(self, user_id, top_k=10):
        """유저에게 제품 추천"""
        if user_id not in self.data_loader.user_to_idx:
            print(f"❌ User {user_id} not found!")
            return []
        
        user_idx = self.data_loader.user_to_idx[user_id]
        
        self.model.eval()
        with torch.no_grad():
            # 모든 제품 예측
            num_products = len(self.data_loader.product_to_idx)
            product_indices = torch.arange(num_products, dtype=torch.long).to(config.DEVICE)
            user_indices = torch.full((num_products,), user_idx, dtype=torch.long).to(config.DEVICE)
            
            predictions = self.model(
                self.graph_data.x,
                self.graph_data.edge_index,
                user_indices,
                product_indices
            )
            
            scores = predictions.cpu().numpy()
        
        # Top-K
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        recommendations = []
        for rank, idx in enumerate(top_indices, 1):
            product_id = self.data_loader.idx_to_product[idx]
            product = self.data_loader.products_df[
                self.data_loader.products_df['product_id'] == product_id
            ].iloc[0]
            
            recommendations.append({
                'rank': rank,
                'product_id': product_id,
                'product_name': product['product_name'],
                'brand': product['brand'],
                'category': product['category'],
                'predicted_rating': float(scores[idx]),
                'ingredients': product['ingredient_list'][:5]
            })
        
        return recommendations


# ============================================================================
# 7. 메인 실행
# ============================================================================
def main():
    """메인 실행 함수"""
    
    print("\n실행을 시작합니다...")
    
    # 1. 데이터 로드
    data_loader = DataLoader(config.PRODUCTS_FILE, config.REVIEWS_FILE)
    data_loader.load().preprocess()
    
    # 2. 그래프 생성
    graph_builder = GraphBuilder(data_loader)
    graph_builder.build_edges().build_features()
    graph_data = graph_builder.get_data()
    
    # 3. 모델 생성
    print(f"\n🤖 [Model] GNN 모델 생성 중...")
    model = RecommendationModel(
        num_products=graph_builder.num_products,
        num_ingredients=graph_builder.num_ingredients,
        num_users=graph_builder.num_users
    ).to(config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ 총 파라미터: {total_params:,}개")
    
    # 4. 학습
    trainer = Trainer(model, graph_data, data_loader)
    history = trainer.train()
    
    # 5. 평가
    evaluator = Evaluator(model, graph_data, trainer.test_reviews, data_loader)
    metrics = evaluator.evaluate()
    
    # 6. 샘플 추천
    print("\n" + "=" * 80)
    print("🎯 [8] 샘플 추천".center(80))
    print("=" * 80)
    
    rec_system = RecommendationSystem(model, graph_data, data_loader)
    
    sample_user = data_loader.reviews_df['user_id'].iloc[0]
    print(f"\n👤 유저: {sample_user}")
    print("-" * 80)
    
    recommendations = rec_system.recommend(sample_user, top_k=5)
    
    for rec in recommendations:
        print(f"\n{rec['rank']}. {rec['product_name']}")
        print(f"   브랜드: {rec['brand']} | 카테고리: {rec['category']}")
        print(f"   예상 평점: {rec['predicted_rating']:.2f}/5.0")
        print(f"   주요 성분: {', '.join(rec['ingredients'][:3])}")
    
    print("\n" + "=" * 80)
    print("✅ 완료!".center(80))
    print("=" * 80)
    
    return model, data_loader, rec_system, metrics


if __name__ == "__main__":
    model, data_loader, rec_system, metrics = main()
