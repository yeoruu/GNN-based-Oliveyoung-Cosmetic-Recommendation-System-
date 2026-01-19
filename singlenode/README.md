# 단일 노드 GNN 기반 화장품 추천 시스템

## 📝 프로젝트 설명
이 프로젝트는 **단일 노드(제품 노드)에 피처를 추가**하는 방식의 GNN을 사용한 화장품 추천 시스템입니다.

### 주요 특징
- **단일 노드 구조**: 제품만을 노드로 사용 (Heterogeneous 아님)
- **풍부한 노드 피처**: 카테고리, 성분, 평점, 피부타입별 선호도 등
- **다양한 엣지**: 성분 유사도, 카테고리, 협업 필터링 기반
- **4가지 평가지표**: AUC, Recall@5, AP, NDCG@5

## 🚀 설치 방법

### 1. 필수 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. PyTorch Geometric 설치 (운영체제별)
```bash
# CUDA가 있는 경우
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# CPU만 사용하는 경우
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

## 📂 파일 구조
```
project/
├── final_products.csv          # 제품 데이터
├── final_total_reviews.csv     # 리뷰 데이터
├── single_node_gnn_recommender.py  # 메인 코드
├── requirements.txt            # 필수 패키지
└── README.md                   # 이 파일
```

## 💻 실행 방법

### 기본 실행 (학습 + 추천)
```bash
python single_node_gnn_recommender.py
```

### 커스텀 실행
```python
from single_node_gnn_recommender import SingleNodeGNNRecommender

# 1. 시스템 초기화
recommender = SingleNodeGNNRecommender(
    products_path='final_products.csv',
    reviews_path='final_total_reviews.csv'
)

# 2. 모델 학습
recommender.train_model(
    epochs=50,          # 에폭 수
    hidden_channels=128, # 은닉층 차원
    lr=0.001            # 학습률
)

# 3. 추천 받기
recommendations = recommender.recommend(
    skin_type='건성',           # 피부타입: 건성, 지성, 복합성, 민감성
    category='로션',            # 카테고리 (None이면 전체)
    favorite_product_id='L1',  # 좋아하는 제품 ID (선택사항)
    top_n=5                    # 추천 개수
)

# 4. 결과 출력
recommender.print_recommendations(recommendations)
```

## 🏗️ 모델 구조

### 노드 피처 (제품 노드)
1. **카테고리 원-핫 인코딩**: 제품 카테고리 정보
2. **성분 벡터**: 모든 성분에 대한 Multi-hot 인코딩
3. **제품 평점**: 공식 평점 (0-1 정규화)
4. **피부타입별 선호도**: 각 피부타입 사용자들의 선호도 분포
5. **평균 사용자 평점**: 실제 사용자들이 준 평균 평점

### 엣지 구성
1. **성분 유사도 기반**: 코사인 유사도 top-K
2. **같은 카테고리**: 동일 카테고리 제품 연결
3. **협업 필터링**: 같은 사용자가 높게 평가한 제품들 연결

### GNN 아키텍처
```
입력 피처 (제품) → GCN Layer 1 → ReLU → Dropout
                  → GCN Layer 2 → ReLU → Dropout
                  → GCN Layer 3
                  
사용자 피처 + 제품 임베딩 → MLP → 예측 점수
```

## 📊 평가 지표

- **AUC (Area Under ROC Curve)**: 이진 분류 성능
- **Recall@5**: Top-5 추천의 재현율
- **AP (Average Precision)**: 정밀도-재현율 곡선 아래 면적
- **NDCG@5**: 순위를 고려한 추천 품질

## 🎯 입력 및 출력

### 입력
- **피부타입**: 건성, 지성, 복합성, 민감성
- **카테고리**: 로션, 세럼, 크림 등 (전체 선택 가능)
- **선호 제품**: 이전에 좋았던 제품 1개 (선택사항)

### 출력 (제품당)
- **제품명**: 화장품 이름
- **브랜드**: 제조사
- **URL**: 제품 상세 페이지 링크
- **주요 성분**: 상위 5개 주요 성분
- **리뷰 3개**: 같은 피부타입 사용자들의 리뷰

## 🔧 하이퍼파라미터 튜닝

```python
# 모델 학습 시 조정 가능한 파라미터
recommender.train_model(
    epochs=100,           # 기본값: 100
    hidden_channels=256,  # 기본값: 128 (64, 128, 256 추천)
    lr=0.0005            # 기본값: 0.001 (0.0001 ~ 0.01)
)
```

## 📈 성능 개선 팁

1. **에폭 수 증가**: 50 → 100으로 증가하면 성능 향상
2. **은닉층 차원 조정**: 128 → 256으로 증가 (메모리 여유 있는 경우)
3. **학습률 조정**: 0.001 → 0.0005로 감소하면 안정적 학습
4. **엣지 수 증가**: `_compute_product_edges`의 `top_k` 파라미터 조정

## 🐛 문제 해결

### CUDA Out of Memory
```python
# 배치 사이즈 감소
# single_node_gnn_recommender.py의 line 358
batch_size = 256  # 기본값: 512
```

### 학습이 너무 느림
```python
# CPU 사용
device = torch.device('cpu')

# 또는 샘플 데이터로 테스트
test_reviews = recommender.reviews_df.sample(frac=0.1)
```

## 📝 코드 주요 함수

### `_build_graph()`
제품 노드와 엣지를 구성하는 핵심 함수
- 제품 피처 생성
- 엣지 연결 (유사도, 카테고리, 협업 필터링)

### `train_model()`
GNN 모델 학습
- MSE Loss 사용
- Adam Optimizer
- 매 10 에폭마다 검증

### `recommend()`
실시간 추천 생성
- 가상 사용자 생성
- 전체 제품에 대해 예측
- Top-N 선택 및 리뷰 매칭

## 🎓 모델 저장 및 로드

```python
# 저장
torch.save({
    'model_state': recommender.model.state_dict(),
    'product_encoder': recommender.product_encoder,
    # ... 기타 인코더
}, 'my_model.pt')

# 로드
checkpoint = torch.load('my_model.pt')
recommender.model.load_state_dict(checkpoint['model_state'])
```

## 📧 문의사항
프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.

## 📄 라이센스
이 프로젝트는 교육 목적으로 제공됩니다.
