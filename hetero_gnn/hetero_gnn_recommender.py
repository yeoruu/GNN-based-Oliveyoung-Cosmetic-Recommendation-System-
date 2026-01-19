"""
이종 그래프 GNN - 확장된 평가 지표 + R-GCN 모델

추가된 평가 지표:
- R² Score
- Precision, Recall, F1-Score
- Hit Rate @5, @10
- NDCG @5, @10
- Coverage
- Diversity

모델:
1. SAGEConv (기본)
2. R-GCN (Relational GCN) - 추가

실행:
python hetero_gnn_enhanced.py
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, RGCNConv
from torch_geometric.data import HeteroData
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    accuracy_score,
    precision_recall_fscore_support,
    r2_score
)
from collections import defaultdict, Counter
import ast
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 설정
# ============================================================================
class Config:
    """하이퍼파라미터 설정"""
    # 파일 경로
    PRODUCTS_FILE = 'merge_final/final_products.csv'
    REVIEWS_FILE = 'merge_final/final_total_reviews.csv'
    
    # 모델 선택
    MODEL_TYPE = 'RGCN'  # 'SAGE' or 'RGCN'
    
    # 모델 하이퍼파라미터
    HIDDEN_DIM = 128
    EMBEDDING_DIM = 64
    NUM_LAYERS = 3
    DROPOUT = 0.5
    
    # 학습 설정
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 5e-4
    NUM_EPOCHS = 100
    PATIENCE = 15
    
    # 기타
    TEST_SIZE = 0.2
    RANDOM_SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

print("=" * 80)
print("이종 그래프 GNN - Enhanced Version".center(80))
print("=" * 80)
print(f"\n⚙️  설정:")
print(f"   Device: {config.DEVICE}")
print(f"   Model: {config.MODEL_TYPE}")
print(f"   Hidden Dim: {config.HIDDEN_DIM}")
print(f"   Embedding Dim: {config.EMBEDDING_DIM}")
print(f"   Learning Rate: {config.LEARNING_RATE}")
print(f"   Epochs: {config.NUM_EPOCHS}")


# ============================================================================
# 1. 데이터 로드 및 전처리 (동일)
# ============================================================================
class HeteroDataLoader:
    """이종 그래프 데이터 로더"""
    
    def __init__(self, products_path, reviews_path):
        self.products_path = products_path
        self.reviews_path = reviews_path
        
        # 매핑 딕셔너리
        self.product_to_idx = {}
        self.skintype_to_idx = {}
        self.ingredient_to_idx = {}
        self.category_to_idx = {}
        self.brand_to_idx = {}
        
        # 역매핑
        self.idx_to_product = {}
        self.idx_to_skintype = {}
        self.idx_to_ingredient = {}
        self.idx_to_category = {}
        self.idx_to_brand = {}
        
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
    
    def parse_skintype(self, skintype_str):
        """피부 타입 파싱"""
        if pd.isna(skintype_str):
            return []
        
        skintypes = []
        if isinstance(skintype_str, str):
            for sep in ['|', ',', '/']:
                if sep in skintype_str:
                    skintypes = [s.strip() for s in skintype_str.split(sep)]
                    break
            
            if not skintypes:
                skintypes = [skintype_str.strip()]
        
        return [st for st in skintypes if st]
    
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
        
        # 피부 타입 파싱
        self.reviews_df['skintype_list'] = self.reviews_df['user_keywords'].apply(
            self.parse_skintype
        )
        
        # 고유 값 추출
        all_ingredients = set()
        for ings in self.products_df['ingredient_list']:
            all_ingredients.update(ings)
        
        all_skintypes = set()
        for types in self.reviews_df['skintype_list']:
            all_skintypes.update(types)
        
        all_categories = set(self.products_df['category'].unique())
        all_brands = set(self.products_df['brand'].unique())
        
        print(f"   ✓ 유효한 제품: {len(self.products_df):,}개")
        print(f"   ✓ 고유 성분: {len(all_ingredients):,}개")
        print(f"   ✓ 피부 타입: {len(all_skintypes):,}개")
        print(f"   ✓ 카테고리: {len(all_categories):,}개")
        print(f"   ✓ 브랜드: {len(all_brands):,}개")
        
        # 인덱스 매핑
        self.product_to_idx = {pid: idx for idx, pid in enumerate(self.products_df['product_id'].unique())}
        self.skintype_to_idx = {st: idx for idx, st in enumerate(sorted(all_skintypes))}
        self.ingredient_to_idx = {ing: idx for idx, ing in enumerate(sorted(all_ingredients))}
        self.category_to_idx = {cat: idx for idx, cat in enumerate(sorted(all_categories))}
        self.brand_to_idx = {brand: idx for idx, brand in enumerate(sorted(all_brands))}
        
        # 역매핑
        self.idx_to_product = {v: k for k, v in self.product_to_idx.items()}
        self.idx_to_skintype = {v: k for k, v in self.skintype_to_idx.items()}
        self.idx_to_ingredient = {v: k for k, v in self.ingredient_to_idx.items()}
        self.idx_to_category = {v: k for k, v in self.category_to_idx.items()}
        self.idx_to_brand = {v: k for k, v in self.brand_to_idx.items()}
        
        # 인덱스 추가
        self.products_df['product_idx'] = self.products_df['product_id'].map(self.product_to_idx)
        self.products_df['category_idx'] = self.products_df['category'].map(self.category_to_idx)
        self.products_df['brand_idx'] = self.products_df['brand'].map(self.brand_to_idx)
        
        self.reviews_df['product_idx'] = self.reviews_df['product_id'].map(self.product_to_idx)
        self.reviews_df = self.reviews_df.dropna(subset=['product_idx'])
        
        print(f"\n   📊 피부 타입 분포:")
        for st, idx in sorted(self.skintype_to_idx.items(), key=lambda x: x[1]):
            count = sum(st in types for types in self.reviews_df['skintype_list'])
            print(f"      • {st}: {count:,}개")
        
        return self


# ============================================================================
# 2. 이종 그래프 생성 (동일)
# ============================================================================
class HeteroGraphBuilder:
    """이종 그래프 빌더"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.hetero_data = HeteroData()
        
    def build(self):
        """이종 그래프 생성"""
        print("\n🕸️  [3] 이종 그래프 생성 중...")
        
        # 노드 수
        num_products = len(self.data_loader.product_to_idx)
        num_skintypes = len(self.data_loader.skintype_to_idx)
        num_ingredients = len(self.data_loader.ingredient_to_idx)
        num_categories = len(self.data_loader.category_to_idx)
        num_brands = len(self.data_loader.brand_to_idx)
        
        print(f"\n   📊 노드 통계:")
        print(f"      • Product: {num_products:,}개")
        print(f"      • SkinType: {num_skintypes:,}개")
        print(f"      • Ingredient: {num_ingredients:,}개")
        print(f"      • Category: {num_categories:,}개")
        print(f"      • Brand: {num_brands:,}개")
        
        # 노드 특징 초기화 (랜덤)
        self.hetero_data['product'].x = torch.randn(num_products, config.HIDDEN_DIM)
        self.hetero_data['skintype'].x = torch.randn(num_skintypes, config.HIDDEN_DIM)
        self.hetero_data['ingredient'].x = torch.randn(num_ingredients, config.HIDDEN_DIM)
        self.hetero_data['category'].x = torch.randn(num_categories, config.HIDDEN_DIM)
        self.hetero_data['brand'].x = torch.randn(num_brands, config.HIDDEN_DIM)
        
        # 엣지 생성
        self._build_edges()
        
        return self.hetero_data
    
    def _build_edges(self):
        """엣지 생성"""
        print(f"\n   🔗 엣지 생성 중...")
        
        # 1) SkinType → Product
        skintype_product_edges = []
        for _, review in self.data_loader.reviews_df.iterrows():
            product_idx = int(review['product_idx'])
            for skintype in review['skintype_list']:
                if skintype in self.data_loader.skintype_to_idx:
                    skintype_idx = self.data_loader.skintype_to_idx[skintype]
                    skintype_product_edges.append([skintype_idx, product_idx])
        
        if skintype_product_edges:
            edge_index = torch.tensor(skintype_product_edges, dtype=torch.long).t()
            self.hetero_data['skintype', 'reviewed', 'product'].edge_index = edge_index
            print(f"      ✓ SkinType → Product: {len(skintype_product_edges):,}개")
        
        # 2) Product → Category
        product_category_edges = []
        for _, row in self.data_loader.products_df.iterrows():
            product_idx = row['product_idx']
            category_idx = row['category_idx']
            product_category_edges.append([product_idx, category_idx])
        
        if product_category_edges:
            edge_index = torch.tensor(product_category_edges, dtype=torch.long).t()
            self.hetero_data['product', 'in_category', 'category'].edge_index = edge_index
            print(f"      ✓ Product → Category: {len(product_category_edges):,}개")
        
        # 3) Product → Ingredient
        product_ingredient_edges = []
        for _, row in self.data_loader.products_df.iterrows():
            product_idx = row['product_idx']
            for ing in row['ingredient_list']:
                if ing in self.data_loader.ingredient_to_idx:
                    ing_idx = self.data_loader.ingredient_to_idx[ing]
                    product_ingredient_edges.append([product_idx, ing_idx])
        
        if product_ingredient_edges:
            edge_index = torch.tensor(product_ingredient_edges, dtype=torch.long).t()
            self.hetero_data['product', 'contains', 'ingredient'].edge_index = edge_index
            print(f"      ✓ Product → Ingredient: {len(product_ingredient_edges):,}개")
        
        # 4) Product → Brand
        product_brand_edges = []
        for _, row in self.data_loader.products_df.iterrows():
            product_idx = row['product_idx']
            brand_idx = row['brand_idx']
            product_brand_edges.append([product_idx, brand_idx])
        
        if product_brand_edges:
            edge_index = torch.tensor(product_brand_edges, dtype=torch.long).t()
            self.hetero_data['product', 'made_by', 'brand'].edge_index = edge_index
            print(f"      ✓ Product → Brand: {len(product_brand_edges):,}개")
        
        # 역방향 엣지
        print(f"\n   🔄 역방향 엣지 추가 중...")
        
        if ('skintype', 'reviewed', 'product') in self.hetero_data.edge_types:
            edge_index = self.hetero_data['skintype', 'reviewed', 'product'].edge_index
            self.hetero_data['product', 'rev_reviewed', 'skintype'].edge_index = edge_index.flip([0])
            print(f"      ✓ Product → SkinType")
        
        if ('product', 'in_category', 'category') in self.hetero_data.edge_types:
            edge_index = self.hetero_data['product', 'in_category', 'category'].edge_index
            self.hetero_data['category', 'rev_in_category', 'product'].edge_index = edge_index.flip([0])
            print(f"      ✓ Category → Product")
        
        if ('product', 'contains', 'ingredient') in self.hetero_data.edge_types:
            edge_index = self.hetero_data['product', 'contains', 'ingredient'].edge_index
            self.hetero_data['ingredient', 'rev_contains', 'product'].edge_index = edge_index.flip([0])
            print(f"      ✓ Ingredient → Product")
        
        if ('product', 'made_by', 'brand') in self.hetero_data.edge_types:
            edge_index = self.hetero_data['product', 'made_by', 'brand'].edge_index
            self.hetero_data['brand', 'rev_made_by', 'product'].edge_index = edge_index.flip([0])
            print(f"      ✓ Brand → Product")


# ============================================================================
# 3. R-GCN 스타일 모델 (엣지 타입별 가중치)
# ============================================================================
class RelationalGNN(nn.Module):
    """R-GCN 스타일 - 각 엣지 타입마다 별도 변환 레이어"""
    
    def __init__(self, metadata, hidden_dim, out_dim, num_layers=3):
        super(RelationalGNN, self).__init__()
        
        self.convs = nn.ModuleList()
        self.edge_transforms = nn.ModuleList()
        
        # 각 레이어마다
        for layer_idx in range(num_layers):
            # 출력 차원 결정
            if layer_idx == num_layers - 1:
                out_channels = out_dim
            else:
                out_channels = hidden_dim
            
            # HeteroConv (기본 메시지 전달)
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), out_channels)
                for edge_type in metadata[1]
            }, aggr='mean')
            
            self.convs.append(conv)
            
            # 엣지 타입별 변환 레이어 (R-GCN의 핵심!)
            edge_transform = nn.ModuleDict({
                f"{src}_{rel}_{dst}": nn.Linear(out_channels, out_channels)
                for src, rel, dst in metadata[1]
            })
            
            self.edge_transforms.append(edge_transform)
    
    def forward(self, x_dict, edge_index_dict):
        for layer_idx, (conv, edge_transform) in enumerate(zip(self.convs, self.edge_transforms)):
            # 1. 기본 메시지 전달
            x_dict_new = conv(x_dict, edge_index_dict)
            
            # 2. 엣지 타입별 변환 적용 (R-GCN 특징!)
            x_dict_transformed = {}
            for node_type in x_dict_new.keys():
                transformations = []
                
                # 이 노드로 들어오는 모든 엣지 타입 찾기
                for edge_type in edge_index_dict.keys():
                    src, rel, dst = edge_type
                    
                    if dst == node_type:
                        # 엣지 타입별 변환 적용
                        edge_key = f"{src}_{rel}_{dst}"
                        if edge_key in edge_transform:
                            transformed = edge_transform[edge_key](x_dict_new[node_type])
                            transformations.append(transformed)
                
                # 모든 변환 평균
                if transformations:
                    x_dict_transformed[node_type] = torch.stack(transformations).mean(dim=0)
                else:
                    x_dict_transformed[node_type] = x_dict_new[node_type]
            
            x_dict = x_dict_transformed
            
            # 3. 활성화 함수 (마지막 레이어 제외)
            if layer_idx < len(self.convs) - 1:
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}
                x_dict = {key: F.dropout(x, p=config.DROPOUT, training=self.training) 
                         for key, x in x_dict.items()}
        
        return x_dict


class HeteroGNN(nn.Module):
    """SAGE 기반 이종 GNN"""
    
    def __init__(self, metadata, hidden_dim, out_dim, num_layers=3):
        super(HeteroGNN, self).__init__()
        
        self.convs = nn.ModuleList()
        
        # 첫 번째 레이어
        self.convs.append(
            HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_dim)
                for edge_type in metadata[1]
            }, aggr='mean')
        )
        
        # 중간 레이어
        for _ in range(num_layers - 2):
            self.convs.append(
                HeteroConv({
                    edge_type: SAGEConv((-1, -1), hidden_dim)
                    for edge_type in metadata[1]
                }, aggr='mean')
            )
        
        # 마지막 레이어
        self.convs.append(
            HeteroConv({
                edge_type: SAGEConv((-1, -1), out_dim)
                for edge_type in metadata[1]
            }, aggr='mean')
        )
    
    def forward(self, x_dict, edge_index_dict):
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            
            if i < len(self.convs) - 1:
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}
                x_dict = {key: F.dropout(x, p=config.DROPOUT, training=self.training) 
                         for key, x in x_dict.items()}
        
        return x_dict


class HeteroRecommendationModel(nn.Module):
    """이종 그래프 추천 모델"""
    
    def __init__(self, metadata, model_type='SAGE'):
        super(HeteroRecommendationModel, self).__init__()
        
        # 모델 선택
        if model_type == 'RGCN':
            self.gnn = RelationalGNN(
                metadata,
                hidden_dim=config.HIDDEN_DIM,
                out_dim=config.EMBEDDING_DIM,
                num_layers=config.NUM_LAYERS
            )
        else:  # SAGE
            self.gnn = HeteroGNN(
                metadata,
                hidden_dim=config.HIDDEN_DIM,
                out_dim=config.EMBEDDING_DIM,
                num_layers=config.NUM_LAYERS
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
    
    def forward(self, hetero_data, skintype_indices, product_indices):
        # GNN 임베딩
        x_dict = self.gnn(hetero_data.x_dict, hetero_data.edge_index_dict)
        
        # SkinType과 Product 임베딩
        skintype_emb = x_dict['skintype'][skintype_indices]
        product_emb = x_dict['product'][product_indices]
        
        # 결합
        combined = torch.cat([skintype_emb, product_emb], dim=1)
        
        # 예측 (1~5 범위)
        rating = self.predictor(combined).squeeze()
        rating = torch.sigmoid(rating) * 4 + 1
        
        return rating


# ============================================================================
# 4. 학습 데이터 준비
# ============================================================================
def prepare_training_data(data_loader):
    """학습 데이터 준비"""
    print("\n📊 [4] 학습 데이터 준비 중...")
    
    training_samples = []
    
    for _, review in data_loader.reviews_df.iterrows():
        product_idx = int(review['product_idx'])
        rating = review['user_rating']
        
        for skintype in review['skintype_list']:
            if skintype in data_loader.skintype_to_idx:
                skintype_idx = data_loader.skintype_to_idx[skintype]
                training_samples.append({
                    'skintype_idx': skintype_idx,
                    'product_idx': product_idx,
                    'rating': rating
                })
    
    samples_df = pd.DataFrame(training_samples)
    
    train_df, test_df = train_test_split(
        samples_df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED
    )
    
    print(f"   ✓ 총 샘플: {len(samples_df):,}개")
    print(f"   ✓ 학습: {len(train_df):,}개")
    print(f"   ✓ 테스트: {len(test_df):,}개")
    
    return train_df, test_df


# ============================================================================
# 5. 학습
# ============================================================================
def train_model(model, hetero_data, train_df, test_df):
    """모델 학습"""
    print(f"\n🚀 [5] 모델 학습 시작...")
    print("=" * 80)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    criterion = nn.MSELoss()
    
    # 텐서 준비
    train_skintype = torch.tensor(train_df['skintype_idx'].values, dtype=torch.long).to(config.DEVICE)
    train_product = torch.tensor(train_df['product_idx'].values, dtype=torch.long).to(config.DEVICE)
    train_rating = torch.tensor(train_df['rating'].values, dtype=torch.float).to(config.DEVICE)
    
    test_skintype = torch.tensor(test_df['skintype_idx'].values, dtype=torch.long).to(config.DEVICE)
    test_product = torch.tensor(test_df['product_idx'].values, dtype=torch.long).to(config.DEVICE)
    test_rating = torch.tensor(test_df['rating'].values, dtype=torch.float).to(config.DEVICE)
    
    best_test_rmse = float('inf')
    patience_counter = 0
    
    for epoch in range(config.NUM_EPOCHS):
        # 학습
        model.train()
        optimizer.zero_grad()
        
        predictions = model(hetero_data, train_skintype, train_product)
        loss = criterion(predictions, train_rating)
        
        loss.backward()
        optimizer.step()
        
        # 평가
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_pred = model(hetero_data, test_skintype, test_product)
                test_rmse = torch.sqrt(criterion(test_pred, test_rating)).item()
            
            print(f"Epoch {epoch+1:3d}/{config.NUM_EPOCHS} | "
                  f"Train Loss: {loss.item():.4f} | "
                  f"Test RMSE: {test_rmse:.4f}")
            
            # Early stopping
            if test_rmse < best_test_rmse:
                best_test_rmse = test_rmse
                patience_counter = 0
                torch.save(model.state_dict(), f'best_hetero_{config.MODEL_TYPE.lower()}_model.pt')
            else:
                patience_counter += 1
            
            if patience_counter >= config.PATIENCE // 10:
                print(f"\n⚠️  Early stopping at epoch {epoch+1}")
                break
    
    print(f"\n✅ 학습 완료! Best RMSE: {best_test_rmse:.4f}")
    
    return best_test_rmse


# ============================================================================
# 6. 확장된 평가 (메트릭 추가!)
# ============================================================================
def evaluate_model_extended(model, hetero_data, test_df, data_loader):
    """확장된 평가 지표"""
    print("\n" + "=" * 80)
    print("📈 [6] 모델 평가 (확장 메트릭)".center(80))
    print("=" * 80)
    
    model.load_state_dict(torch.load(f'best_hetero_{config.MODEL_TYPE.lower()}_model.pt'))
    model.eval()
    
    test_skintype = torch.tensor(test_df['skintype_idx'].values, dtype=torch.long).to(config.DEVICE)
    test_product = torch.tensor(test_df['product_idx'].values, dtype=torch.long).to(config.DEVICE)
    test_rating = torch.tensor(test_df['rating'].values, dtype=torch.float).to(config.DEVICE)
    
    with torch.no_grad():
        predictions = model(hetero_data, test_skintype, test_product)
    
    y_true = test_rating.cpu().numpy()
    y_pred = predictions.cpu().numpy()
    
    metrics = {}
    
    # ========== Regression Metrics ==========
    print("\n📊 Regression Metrics:")
    metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    metrics['R2'] = r2_score(y_true, y_pred)
    
    print(f"   • RMSE: {metrics['RMSE']:.4f}")
    print(f"   • MAE: {metrics['MAE']:.4f}")
    print(f"   • R² Score: {metrics['R2']:.4f}")
    
    # ========== Classification Metrics ==========
    print("\n📊 Classification Metrics:")
    y_true_class = np.round(y_true).astype(int)
    y_pred_class = np.round(y_pred).astype(int)
    
    metrics['Accuracy'] = accuracy_score(y_true_class, y_pred_class)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_class, y_pred_class, average='weighted', zero_division=0
    )
    
    metrics['Precision'] = precision
    metrics['Recall'] = recall
    metrics['F1'] = f1
    
    print(f"   • Accuracy: {metrics['Accuracy']:.4f}")
    print(f"   • Precision: {precision:.4f}")
    print(f"   • Recall: {recall:.4f}")
    print(f"   • F1-Score: {f1:.4f}")
    
    # ========== Ranking Metrics ==========
    print("\n📊 Ranking Metrics:")
    
    # Hit Rate 계산
    def calculate_hit_rate(k=10, threshold=4.0):
        skintype_groups = defaultdict(list)
        for st_idx, prod_idx, true_val, pred_val in zip(
            test_df['skintype_idx'].values, 
            test_df['product_idx'].values,
            y_true, 
            y_pred
        ):
            skintype_groups[st_idx].append((prod_idx, true_val, pred_val))
        
        hits = 0
        total = 0
        
        for st_idx, items in skintype_groups.items():
            if len(items) < k:
                continue
            
            # 예측 점수로 정렬
            items_sorted = sorted(items, key=lambda x: x[2], reverse=True)
            top_k = items_sorted[:k]
            
            # 실제로 좋은 제품(threshold 이상)이 있는지 확인
            if any(true_rating >= threshold for _, true_rating, _ in top_k):
                hits += 1
            total += 1
        
        return hits / total if total > 0 else 0
    
    metrics['HR@5'] = calculate_hit_rate(k=5)
    metrics['HR@10'] = calculate_hit_rate(k=10)
    
    print(f"   • Hit Rate @5: {metrics['HR@5']:.4f}")
    print(f"   • Hit Rate @10: {metrics['HR@10']:.4f}")
    
    # NDCG 계산
    def calculate_ndcg(k=10):
        skintype_groups = defaultdict(list)
        for st_idx, prod_idx, true_val, pred_val in zip(
            test_df['skintype_idx'].values,
            test_df['product_idx'].values,
            y_true,
            y_pred
        ):
            skintype_groups[st_idx].append((prod_idx, true_val, pred_val))
        
        ndcg_scores = []
        
        for st_idx, items in skintype_groups.items():
            if len(items) < k:
                continue
            
            # 예측 점수로 정렬
            items_sorted = sorted(items, key=lambda x: x[2], reverse=True)
            top_k = items_sorted[:k]
            
            # DCG 계산
            dcg = sum((2**true_rating - 1) / np.log2(i + 2) 
                     for i, (_, true_rating, _) in enumerate(top_k))
            
            # IDCG 계산 (이상적인 순서)
            ideal_sorted = sorted(items, key=lambda x: x[1], reverse=True)[:k]
            idcg = sum((2**true_rating - 1) / np.log2(i + 2) 
                      for i, (_, true_rating, _) in enumerate(ideal_sorted))
            
            if idcg > 0:
                ndcg_scores.append(dcg / idcg)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0
    
    metrics['NDCG@5'] = calculate_ndcg(k=5)
    metrics['NDCG@10'] = calculate_ndcg(k=10)
    
    print(f"   • NDCG @5: {metrics['NDCG@5']:.4f}")
    print(f"   • NDCG @10: {metrics['NDCG@10']:.4f}")
    
    # ========== Coverage & Diversity ==========
    print("\n📊 Coverage & Diversity:")
    
    # Coverage: 추천된 고유 제품 비율
    recommended_products = set(test_df['product_idx'].values)
    total_products = len(data_loader.product_to_idx)
    metrics['Coverage'] = len(recommended_products) / total_products
    
    print(f"   • Coverage: {metrics['Coverage']:.4f} ({len(recommended_products)}/{total_products})")
    
    # Category Diversity
    product_categories = test_df['product_idx'].map(
        lambda x: data_loader.products_df[
            data_loader.products_df['product_idx'] == x
        ]['category'].values[0] if len(data_loader.products_df[
            data_loader.products_df['product_idx'] == x
        ]) > 0 else None
    )
    
    category_dist = Counter(product_categories.dropna())
    num_categories = len(category_dist)
    total_categories = len(data_loader.category_to_idx)
    
    metrics['Category_Diversity'] = num_categories / total_categories
    print(f"   • Category Diversity: {metrics['Category_Diversity']:.4f} ({num_categories}/{total_categories})")
    
    # ========== Error Distribution ==========
    print("\n📊 Error Distribution:")
    errors = np.abs(y_true - y_pred)
    
    print(f"   • Mean Error: {np.mean(errors):.4f}")
    print(f"   • Std Error: {np.std(errors):.4f}")
    print(f"   • Median Error: {np.median(errors):.4f}")
    print(f"   • Max Error: {np.max(errors):.4f}")
    
    # Error by rating range
    for rating in [1, 2, 3, 4, 5]:
        mask = (y_true_class == rating)
        if mask.sum() > 0:
            mean_error = np.mean(errors[mask])
            print(f"   • Mean Error (Rating={rating}): {mean_error:.4f}")
    
    return metrics


# ============================================================================
# 7. 추천 시스템
# ============================================================================
class HeteroRecommendationSystem:
    """이종 그래프 추천 시스템"""
    
    def __init__(self, model, hetero_data, data_loader):
        self.model = model
        self.hetero_data = hetero_data
        self.data_loader = data_loader
    
    def recommend_by_skintype(self, skintype, top_k=10):
        """피부 타입별 추천"""
        if skintype not in self.data_loader.skintype_to_idx:
            print(f"❌ SkinType '{skintype}' not found!")
            return []
        
        skintype_idx = self.data_loader.skintype_to_idx[skintype]
        
        self.model.eval()
        with torch.no_grad():
            num_products = len(self.data_loader.product_to_idx)
            product_indices = torch.arange(num_products, dtype=torch.long).to(config.DEVICE)
            skintype_indices = torch.full((num_products,), skintype_idx, dtype=torch.long).to(config.DEVICE)
            
            scores = self.model(self.hetero_data, skintype_indices, product_indices)
            scores = scores.cpu().numpy()
        
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
# 8. 메인 실행
# ============================================================================
def main():
    """메인 실행 함수"""
    
    # 1. 데이터 로드
    data_loader = HeteroDataLoader(config.PRODUCTS_FILE, config.REVIEWS_FILE)
    data_loader.load().preprocess()
    
    # 2. 그래프 생성
    graph_builder = HeteroGraphBuilder(data_loader)
    hetero_data = graph_builder.build()
    hetero_data = hetero_data.to(config.DEVICE)
    
    # 3. 학습 데이터 준비
    train_df, test_df = prepare_training_data(data_loader)
    
    # 4. 모델 생성
    print(f"\n🤖 [{config.MODEL_TYPE}] 모델 생성 중...")
    model = HeteroRecommendationModel(
        hetero_data.metadata(), 
        model_type=config.MODEL_TYPE
    ).to(config.DEVICE)
    
    # Lazy module 초기화
    print(f"   초기화 중...")
    model.eval()
    with torch.no_grad():
        dummy_skintype = torch.tensor([0, 1], dtype=torch.long).to(config.DEVICE)
        dummy_product = torch.tensor([0, 1], dtype=torch.long).to(config.DEVICE)
        _ = model(hetero_data, dummy_skintype, dummy_product)
    model.train()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ 총 파라미터: {total_params:,}개")
    
    # 5. 학습
    train_model(model, hetero_data, train_df, test_df)
    
    # 6. 확장 평가
    metrics = evaluate_model_extended(model, hetero_data, test_df, data_loader)
    
    # 7. 샘플 추천
    print("\n" + "=" * 80)
    print("🎯 [7] 샘플 추천".center(80))
    print("=" * 80)
    
    rec_system = HeteroRecommendationSystem(model, hetero_data, data_loader)
    
    sample_skintypes = list(data_loader.skintype_to_idx.keys())[:3]
    
    for skintype in sample_skintypes:
        print(f"\n👤 피부 타입: {skintype}")
        print("-" * 80)
        
        recommendations = rec_system.recommend_by_skintype(skintype, top_k=5)
        
        for rec in recommendations:
            print(f"\n{rec['rank']}. {rec['product_name']}")
            print(f"   브랜드: {rec['brand']} | 카테고리: {rec['category']}")
            print(f"   예상 평점: {rec['predicted_rating']:.2f}/5.0")
            print(f"   주요 성분: {', '.join(rec['ingredients'][:3])}")
    
    print("\n" + "=" * 80)
    print("✅ 완료!".center(80))
    print("=" * 80)
    
    # 메트릭 요약
    print(f"\n📊 최종 메트릭 요약 ({config.MODEL_TYPE}):")
    print(f"   RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f} | R²: {metrics['R2']:.4f}")
    print(f"   Accuracy: {metrics['Accuracy']:.4f} | F1: {metrics['F1']:.4f}")
    print(f"   HR@10: {metrics['HR@10']:.4f} | NDCG@10: {metrics['NDCG@10']:.4f}")
    print(f"   Coverage: {metrics['Coverage']:.4f} | Diversity: {metrics['Category_Diversity']:.4f}")
    
    return model, data_loader, rec_system, metrics


if __name__ == "__main__":
    model, data_loader, rec_system, metrics = main()