# 🚀 PyTorch GNN 추천 시스템 실행 가이드

## 📋 준비 사항

### 1. Python 버전
- Python 3.8 이상 필요

### 2. 라이브러리 설치

#### 옵션 A: pip 사용
```bash
pip install -r requirements.txt
```

#### 옵션 B: 개별 설치
```bash
# PyTorch (CPU 버전)
pip install torch torchvision torchaudio

# PyTorch (GPU 버전 - CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# PyTorch Geometric
pip install torch-geometric

# 기타 라이브러리
pip install scikit-learn pandas numpy
```

### 3. 데이터 파일 확인
다음 파일들이 필요합니다:
- `final_products.csv` (제품 데이터)
- `final_total_reviews.csv` (리뷰 데이터)

## 🎯 실행 방법

### 기본 실행
```bash
python gnn_recommender_pytorch.py
```

### 코드 내 설정 변경

`gnn_recommender_pytorch.py` 파일의 `Config` 클래스에서 설정을 변경할 수 있습니다:

```python
class Config:
    # 파일 경로 (실제 경로로 변경)
    PRODUCTS_FILE = 'final_products.csv'  # 수정 필요
    REVIEWS_FILE = 'final_total_reviews.csv'  # 수정 필요
    
    # 모델 크기
    FEATURE_DIM = 64        # 작게: 32, 크게: 128
    HIDDEN_DIM = 128        # 작게: 64, 크게: 256
    EMBEDDING_DIM = 64
    
    # 학습 설정
    LEARNING_RATE = 0.001   # 더 빠르게: 0.01, 더 안정적: 0.0001
    NUM_EPOCHS = 150        # 빠르게 테스트: 50, 충분히: 200
    PATIENCE = 20           # Early stopping
    
    # GPU/CPU
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## 📊 예상 실행 시간

### CPU 환경
- 데이터 로드: ~5초
- 그래프 생성: ~10초
- 모델 학습 (150 epochs): ~30-60분
- 평가: ~5초

### GPU 환경 (권장)
- 데이터 로드: ~5초
- 그래프 생성: ~10초
- 모델 학습 (150 epochs): ~5-10분
- 평가: ~2초

## 💾 출력 파일

실행 후 생성되는 파일:
- `best_gnn_model.pt` - 최고 성능 모델 (자동 저장)

## 📈 예상 출력

```
================================================================================
                         PyTorch GNN 기반 화장품 추천 시스템                        
================================================================================

⚙️  설정:
   Device: cuda (또는 cpu)
   Feature Dim: 64
   Hidden Dim: 128
   Learning Rate: 0.001
   Epochs: 150

📂 [1] 데이터 로드 중...
   ✓ 제품: 611개
   ✓ 리뷰: 61,100개

🔧 [2] 데이터 전처리 중...
   ✓ 유효한 제품: 607개
   ✓ 고유 성분: 1,682개
   ✓ 사용자: 18,958명
   ✓ 처리된 리뷰: 60,700개

🕸️  [3] 그래프 엣지 생성 중...
   ✓ Product-Ingredient 엣지: 36,040개
   ✓ User-Product 엣지: 121,400개
   ✓ 총 엣지: 157,440개

🎨 [4] 노드 특징 생성 중...
   ✓ 노드 특징: torch.Size([21247, 64])

📊 [5] 데이터 분할:
   Train: 48,560개
   Test: 12,140개

🤖 [Model] GNN 모델 생성 중...
   ✓ 총 파라미터: 234,567개

🚀 [6] 모델 학습 시작 (Epochs: 150)...
================================================================================
Epoch  10/150 | Train Loss: 0.8234 | Test RMSE: 0.4567
Epoch  20/150 | Train Loss: 0.6123 | Test RMSE: 0.3876
Epoch  30/150 | Train Loss: 0.4567 | Test RMSE: 0.3245
...
Epoch 100/150 | Train Loss: 0.2345 | Test RMSE: 0.2876

✅ 학습 완료! Best RMSE: 0.2876

================================================================================
                                 📈 [7] 모델 평가                                
================================================================================

📊 Regression Metrics:
   • RMSE: 0.2876
   • MAE: 0.1234
   • R²: 0.8765

📊 Classification Metrics:
   • Accuracy: 0.8456
   • Precision: 0.8567
   • Recall: 0.8234
   • F1-Score: 0.8398

📊 Ranking Metrics:
   • Hit Rate @5: 0.7234
   • Hit Rate @10: 0.8456

================================================================================
                                🎯 [8] 샘플 추천                                
================================================================================

👤 유저: 제리공쥬
────────────────────────────────────────────────────────────────────────────────

1. 바이옴 베리어 크림 미스트
   브랜드: 유이크 | 카테고리: 미스트/오일
   예상 평점: 4.87/5.0
   주요 성분: 쿠티박테리움, 글리세린, 나이아신아마이드

2. 복숭아 70 나이아신아마이드 세럼
   브랜드: 아누아 | 카테고리: 에센스/세럼/앰플
   예상 평점: 4.82/5.0
   주요 성분: 복숭아수, 글리세린, 나이아신아마이드

...

================================================================================
                                    ✅ 완료!                                    
================================================================================
```

## 🐛 문제 해결

### 1. CUDA out of memory
```python
# Config 클래스에서:
HIDDEN_DIM = 64  # 128에서 줄이기
FEATURE_DIM = 32  # 64에서 줄이기
```

또는 CPU로 강제 실행:
```python
DEVICE = torch.device('cpu')
```

### 2. 파일을 찾을 수 없음
```python
# Config 클래스에서 전체 경로 지정:
PRODUCTS_FILE = '/full/path/to/final_products.csv'
REVIEWS_FILE = '/full/path/to/final_total_reviews.csv'
```

### 3. 메모리 부족
```bash
# 배치 처리 대신 미니배치 사용하도록 코드 수정 필요
# 또는 더 작은 샘플로 테스트
```

### 4. PyTorch Geometric 설치 오류
```bash
# 수동 설치
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install torch-geometric
```

## 🎓 고급 사용법

### 모델 커스터마이징

#### GNN 아키텍처 변경
```python
# GCN 대신 GAT 사용
from torch_geometric.nn import GATConv

class IngredientGNN(nn.Module):
    def __init__(self, ...):
        # GATConv로 교체
        self.convs.append(GATConv(in_channels, hidden_channels, heads=4))
```

#### 학습률 스케줄러 추가
```python
# Trainer 클래스에:
self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    self.optimizer, mode='min', factor=0.5, patience=5
)

# train() 함수에서:
self.scheduler.step(test_rmse)
```

### 저장된 모델 사용

```python
# 모델 로드
model = RecommendationModel(num_products, num_ingredients, num_users)
model.load_state_dict(torch.load('best_gnn_model.pt'))
model.eval()

# 추천 생성
rec_system = RecommendationSystem(model, graph_data, data_loader)
recommendations = rec_system.recommend('user_id', top_k=10)
```

## 📚 참고 자료

- PyTorch 공식 문서: https://pytorch.org/docs/
- PyTorch Geometric 공식 문서: https://pytorch-geometric.readthedocs.io/
- GNN 튜토리얼: https://distill.pub/2021/gnn-intro/

## 💡 팁

1. **처음 실행시**: NUM_EPOCHS=10으로 설정하여 빠르게 테스트
2. **GPU 사용**: 학습 속도가 5-10배 빠름
3. **Early Stopping**: PATIENCE 값으로 과적합 방지
4. **하이퍼파라미터 튜닝**: Grid Search로 최적값 찾기

## ✅ 체크리스트

실행 전:
- [ ] Python 3.8+ 설치
- [ ] PyTorch 설치
- [ ] PyTorch Geometric 설치
- [ ] 데이터 파일 준비
- [ ] 파일 경로 확인

실행 후:
- [ ] best_gnn_model.pt 생성 확인
- [ ] RMSE < 0.5 달성
- [ ] 추천 결과 확인

---

**마지막 업데이트**: 2026년 1월 18일
**지원**: 문제 발생시 Config 설정 확인
