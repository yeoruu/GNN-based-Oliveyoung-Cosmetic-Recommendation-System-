"""
단일 노드(제품) GNN 기반 화장품 추천시스템 (BPR + Negative Sampling + Skin->Ingredient Profile)

- 노드(제품) 피처: 카테고리 one-hot + 브랜드 one-hot + 성분 TF-IDF -> SVD + 피부타입별 평균평점(결측은 global mean으로)
- 엣지: 성분 임베딩(SVD) 코사인 유사도 기반 + 같은 카테고리
- 학습: BPR loss (pos: rating>=4, neg: user가 train에서 안 본 아이템 랜덤)
- 추천/평가: GNN 점수 + (피부타입 성분취향 점수) 하이브리드
  * 리뷰 텍스트/감성/길이 안 씀. 오직 피부타입+평점만 사용.
"""

import os
import ast
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib


# -------------------------
# 전역 함수 (pickle 안전) ✅ lambda 금지
# -------------------------
def identity(x):
    return x


# -------------------------
# 고정 카테고리 5개
# -------------------------
ALLOWED_CATEGORIES = [
    "스킨/토너",
    "로션",
    "에센스/세럼/앰플",
    "크림",
    "미스트/오일"
]

SKIN_TYPES_ALL = ["건성", "민감성", "복합성", "약건성", "지성", "트러블성", "중성"]
SKIN_TO_IDX = {s: i for i, s in enumerate(SKIN_TYPES_ALL)}


# =========================
# 모델
# =========================
class ProductGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, num_layers=3, user_feature_dim=8):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

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
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.3, training=self.training)
        return x

    def score(self, x, edge_index, user_features, product_indices):
        """
        user_features: [B, 8]
        product_indices: [B]
        return: [B] raw scores
        """
        prod_emb = self.forward(x, edge_index)               # [N, hidden]
        sel = prod_emb[product_indices]                      # [B, hidden]
        combined = torch.cat([user_features, sel], dim=-1)   # [B, 8+hidden]
        return self.predictor(combined).squeeze(-1)          # [B]


# =========================
# 추천 시스템
# =========================
class SingleNodeGNNRecommender:
    def __init__(self, products_path, reviews_path, svd_dim=100, top_k_edges=15, debug=False):
        self.debug = debug
        self.svd_dim = svd_dim
        self.top_k_edges = top_k_edges

        # 하이브리드 가중치 (GNN vs 성분취향)
        self.alpha = 0.7  # 0.7*GNN + 0.3*skin->ingredient (원하면 바꿔)

        print("📚 데이터 로딩 중...")
        self.products_df = pd.read_csv(products_path)
        self.reviews_df = pd.read_csv(reviews_path, encoding="utf-8-sig")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  디바이스: {self.device}")

        # encoders
        self.product_encoder = LabelEncoder()
        self.category_encoder = LabelEncoder()
        self.brand_encoder = LabelEncoder()
        self.user_encoder = LabelEncoder()
        self.mlb_ingredients = MultiLabelBinarizer()

        self._preprocess_data()
        self._build_graph()

        # 피부타입 성분 취향 벡터는 train split 이후에 만들 수 있어서,
        # train_bpr 안에서 만들어줄 거임.

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

        self.products_df["ingredients_list"] = self.products_df["ingredients"].apply(parse_ingredients)

        # 스킨타입 결측 처리
        if self.reviews_df["user_keywords"].isna().any():
            na_count = int(self.reviews_df["user_keywords"].isna().sum())
            print(f"⚠️  피부타입 결측값 {na_count}개 -> '알 수 없음'")
            self.reviews_df["user_keywords"] = self.reviews_df["user_keywords"].fillna("알 수 없음")

        self.reviews_df["skin_types"] = self.reviews_df["user_keywords"].apply(
            lambda x: [t.strip() for t in str(x).split("|")]
        )
        self.reviews_df["rating_normalized"] = self.reviews_df["user_rating"] / 5.0

        # 인코딩
        self.products_df["product_encoded"] = self.product_encoder.fit_transform(self.products_df["product_id"])
        self.products_df["category_encoded"] = self.category_encoder.fit_transform(self.products_df["category"])
        self.products_df["brand_encoded"] = self.brand_encoder.fit_transform(self.products_df["brand"])
        self.reviews_df["user_encoded"] = self.user_encoder.fit_transform(self.reviews_df["user_id"])

        product_to_encoded = dict(zip(self.products_df["product_id"], self.products_df["product_encoded"]))
        self.reviews_df["product_encoded"] = self.reviews_df["product_id"].map(product_to_encoded)

        # 성분 vocab 확인용 (멀티핫)
        all_ingredients = self.products_df["ingredients_list"].tolist()
        self.ingredient_multihot = self.mlb_ingredients.fit_transform(all_ingredients)

        # ✅ 성분 TF-IDF (lambda 없이!)
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=identity,
            preprocessor=identity,
            token_pattern=None,
            lowercase=False
        )
        self.ingredient_tfidf = self.tfidf_vectorizer.fit_transform(self.products_df["ingredients_list"])

        # ✅ TF-IDF -> SVD (노드 피처 + 성분 유사도)
        self.svd = TruncatedSVD(n_components=self.svd_dim, random_state=42)
        ing_svd = self.svd.fit_transform(self.ingredient_tfidf)     # (N, svd_dim)
        self.ingredient_svd = normalize(ing_svd)                    # (N, svd_dim), cosine=dot

        print(f"✅ 제품 수: {len(self.product_encoder.classes_)}")
        print(f"✅ 사용자 수: {len(self.user_encoder.classes_)}")
        print(f"✅ 리뷰 수: {len(self.reviews_df)}")
        print(f"✅ 카테고리 수: {len(self.category_encoder.classes_)}")
        print(f"✅ 브랜드 수: {len(self.brand_encoder.classes_)}")
        print(f"✅ 성분 vocab 수: {self.ingredient_multihot.shape[1]}")
        print(f"✅ 성분 TF-IDF shape: {self.ingredient_tfidf.shape}")
        print(f"✅ 성분 SVD shape: {self.ingredient_svd.shape}")

        # skin_type_to_vector (원핫)
        self.skin_type_to_vector = {st: [1.0 if x == st else 0.0 for x in SKIN_TYPES_ALL] for st in SKIN_TYPES_ALL}

    # ---------- 그래프 ----------
    def _build_graph(self):
        print("🔨 그래프 구축 중...")

        # category one-hot
        num_categories = len(self.category_encoder.classes_)
        category_onehot = np.eye(num_categories)[self.products_df["category_encoded"].values]

        # brand one-hot
        num_brands = len(self.brand_encoder.classes_)
        brand_onehot = np.eye(num_brands)[self.products_df["brand_encoded"].values]

        # ✅ skin-type별 제품 평균 평점 (결측은 1.0 말고 global mean으로 채움)
        global_mean_by_skin = {}
        for st in SKIN_TYPES_ALL:
            st_rows = self.reviews_df[self.reviews_df["skin_types"].apply(lambda x: st in x)]
            if len(st_rows) > 0:
                global_mean_by_skin[st] = float(st_rows["user_rating"].mean() / 5.0)
            else:
                global_mean_by_skin[st] = 0.6  # 안전빵(=3점) 느낌

        skin_ratings = np.zeros((len(self.products_df), len(SKIN_TYPES_ALL)), dtype=np.float32)
        for idx, product_id in enumerate(self.products_df["product_id"].values):
            pr = self.reviews_df[self.reviews_df["product_id"] == product_id]
            for i, st in enumerate(SKIN_TYPES_ALL):
                st_reviews = pr[pr["skin_types"].apply(lambda x: st in x)]
                if len(st_reviews) > 0:
                    skin_ratings[idx, i] = float(st_reviews["user_rating"].mean() / 5.0)
                else:
                    # ✅ 결측은 global mean으로
                    skin_ratings[idx, i] = float(global_mean_by_skin[st])

        # ✅ 노드 피처
        node_features = np.concatenate([category_onehot, brand_onehot, self.ingredient_svd, skin_ratings], axis=1)
        self.node_features = torch.FloatTensor(node_features)

        # 엣지 생성
        print("🔗 제품 간 유사도 기반 엣지 생성 중...")
        edges = self._compute_edges(top_k=self.top_k_edges)
        self.edge_index = torch.LongTensor(edges)

        self.data = Data(x=self.node_features, edge_index=self.edge_index)
        print("✅ 그래프 구축 완료")
        print(f"   - 노드 수: {self.data.x.size(0)}")
        print(f"   - 피처 차원: {self.data.x.size(1)}")
        print(f"   - 엣지 수: {self.data.edge_index.size(1)}")

    def _compute_edges(self, top_k=15):
        sim = self.ingredient_svd @ self.ingredient_svd.T  # (N,N)
        n = sim.shape[0]
        edges = []

        # 1) 성분 유사도 top-k
        for i in range(n):
            s = sim[i].copy()
            s[i] = -1
            top = np.argsort(s)[-top_k:]
            for j in top:
                if s[j] > 0.1:
                    edges.append((i, j))

        # 2) 같은 카테고리 약하게 연결
        for cat in self.products_df["category"].unique():
            ids = self.products_df[self.products_df["category"] == cat]["product_encoded"].values
            if len(ids) > 1:
                for a in range(len(ids)):
                    for b in range(a + 1, min(a + 6, len(ids))):
                        edges.append((ids[a], ids[b]))
                        edges.append((ids[b], ids[a]))

        edges = list(set(edges))
        if len(edges) == 0:
            edges = [(i, i + 1) for i in range(n - 1)] + [(i + 1, i) for i in range(n - 1)]

        return np.array(edges).T

    # ---------- 유저 프로필 ----------
    def _build_user_profiles(self, df_train):
        """
        user_encoded -> 8차원 (skin 7 + mean rating 1)
        """
        user_feat = {}
        grouped = df_train.groupby("user_encoded")

        for u, g in grouped:
            vec = np.zeros(7, dtype=np.float32)
            for skins in g["skin_types"].values:
                for st in skins:
                    if st in SKIN_TO_IDX:
                        vec[SKIN_TO_IDX[st]] += 1.0

            s = vec.sum()
            if s > 0:
                vec = vec / s
            else:
                vec = np.ones(7, dtype=np.float32) / 7.0

            r = float(g["rating_normalized"].mean())
            if np.isnan(r):
                r = 0.8

            user_feat[int(u)] = np.concatenate([vec, [r]]).astype(np.float32)
        return user_feat

    # =========================
    # ✅ 피부타입 성분 취향 벡터 (메타패스 핵심!)
    # =========================
    def _build_skin_pref_vectors(self, df_train, pos_threshold=4):
        """
        skin -> ingredient preference vector (svd_dim)
        - df_train에서 피부타입 st 포함 & rating>=pos_threshold 인 제품들의 ingredient_svd 가중합
        - 가중치는 (rating-3) 사용 (4점=1, 5점=2)
        - normalize해서 cosine 비교 가능하게 함
        """
        skin_pref = np.zeros((len(SKIN_TYPES_ALL), self.svd_dim), dtype=np.float32)

        # 긍정 샘플만
        pos = df_train[df_train["user_rating"] >= pos_threshold].copy()

        for _, row in pos.iterrows():
            p = int(row["product_encoded"])
            r = float(row["user_rating"])
            w = max(0.0, r - 3.0)  # 4->1, 5->2
            if w == 0:
                continue

            skins = row["skin_types"]
            for st in skins:
                if st in SKIN_TO_IDX:
                    skin_pref[SKIN_TO_IDX[st]] += (w * self.ingredient_svd[p]).astype(np.float32)

        # normalize (행 단위)
        skin_pref = normalize(skin_pref)
        self.skin_pref_matrix = skin_pref  # (7, svd_dim)

        # 단일 skin_type 바로 뽑게 dict도
        self.skin_pref_vec = {st: self.skin_pref_matrix[SKIN_TO_IDX[st]] for st in SKIN_TYPES_ALL}

        print("✅ 피부타입 성분 취향 벡터 구축 완료:", self.skin_pref_matrix.shape)

    # =========================
    # ✅ BPR 학습
    # =========================
    def train_bpr(self, epochs=50, hidden_channels=128, lr=0.001,
                  batch_size=1024, steps_per_epoch=200, pos_threshold=4):
        print(f"\n🎓 BPR 학습 시작 (epochs={epochs}, steps/epoch={steps_per_epoch})")

        train_reviews, test_reviews = train_test_split(self.reviews_df, test_size=0.2, random_state=42)
        self.train_reviews = train_reviews.reset_index(drop=True)
        self.test_reviews = test_reviews.reset_index(drop=True)

        # ✅ 여기서 피부타입 성분 취향 벡터 만들기 (리뷰텍스트 안씀)
        self._build_skin_pref_vectors(self.train_reviews, pos_threshold=pos_threshold)

        in_channels = self.node_features.size(1)
        self.model = ProductGNN(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=3).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)

        self.data = self.data.to(self.device)

        user_profiles = self._build_user_profiles(self.train_reviews)

        user_seen = (
            self.train_reviews.groupby("user_encoded")["product_encoded"]
            .apply(lambda s: set(s.dropna().astype(int).tolist()))
            .to_dict()
        )

        train_pos = self.train_reviews[self.train_reviews["user_rating"] >= pos_threshold]
        user_pos = (
            train_pos.groupby("user_encoded")["product_encoded"]
            .apply(lambda s: list(set(s.dropna().astype(int).tolist())))
            .to_dict()
        )

        n_items = len(self.products_df)
        users = []
        for u in user_pos.keys():
            seen = user_seen.get(u, set())
            if len(user_pos[u]) > 0 and len(seen) < n_items:
                users.append(int(u))

        if len(users) == 0:
            raise RuntimeError("BPR 학습 불가: positive 유저가 없거나, 모든 아이템을 본 유저뿐임.")

        best_recall = -1.0
        os.makedirs("singlenode", exist_ok=True)

        for epoch in range(1, epochs + 1):
            self.model.train()
            losses = []

            for _ in range(steps_per_epoch):
                bu = np.random.choice(users, size=min(batch_size, len(users)), replace=False)

                pos_items, neg_items, u_feats = [], [], []
                for u in bu:
                    p = int(np.random.choice(user_pos[u]))
                    seen = user_seen.get(u, set())

                    # neg: seen에 없는 아이템 샘플
                    while True:
                        n = int(np.random.randint(0, n_items))
                        if n not in seen:
                            break

                    pos_items.append(p)
                    neg_items.append(n)
                    u_feats.append(user_profiles[u])

                u_feats = torch.FloatTensor(np.stack(u_feats)).to(self.device)
                pos_idx = torch.LongTensor(pos_items).to(self.device)
                neg_idx = torch.LongTensor(neg_items).to(self.device)

                optimizer.zero_grad()
                s_pos = self.model.score(self.data.x, self.data.edge_index, u_feats, pos_idx)
                s_neg = self.model.score(self.data.x, self.data.edge_index, u_feats, neg_idx)

                loss = -torch.log(torch.sigmoid(s_pos - s_neg) + 1e-8).mean()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            if epoch % 5 == 0:
                metrics = self.evaluate(k=5, max_users=1500)
                print(f"Epoch {epoch}/{epochs} | BPR Loss: {np.mean(losses):.4f} | "
                      f"AUC: {metrics['AUC']:.4f} | Recall@5: {metrics['Recall@5']:.4f} | "
                      f"AP: {metrics['AP']:.4f} | NDCG@5: {metrics['NDCG@5']:.4f}")

                if metrics["Recall@5"] > best_recall:
                    best_recall = metrics["Recall@5"]
                    torch.save(self.model.state_dict(), "singlenode/best_single_gnn_model_bpr.pt")

        print(f"\n✅ BPR 학습 완료! best Recall@5={best_recall:.4f}")
        self.model.load_state_dict(torch.load("singlenode/best_single_gnn_model_bpr.pt", map_location=self.device))

    # =========================
    # ✅ 진짜 추천 평가 (하이브리드 스코어 반영)
    # =========================
    def evaluate(self, k=5, max_users=2000, seed=42):
        assert hasattr(self, "model"), "모델 학습 후 evaluate 호출해야 함"
        assert hasattr(self, "train_reviews") and hasattr(self, "test_reviews"), "train_bpr 먼저 돌려야 함"
        assert hasattr(self, "skin_pref_matrix"), "skin_pref_matrix가 없음 (train_bpr에서 생성됨)"

        train_reviews = self.train_reviews
        test_reviews = self.test_reviews

        rng = np.random.default_rng(seed)
        self.model.eval()

        user_train_seen = (
            train_reviews.groupby("user_encoded")["product_encoded"]
            .apply(lambda s: set(s.dropna().astype(int).tolist()))
            .to_dict()
        )

        test_pos = test_reviews[test_reviews["user_rating"] >= 4].copy()
        user_test_pos = (
            test_pos.groupby("user_encoded")["product_encoded"]
            .apply(lambda s: set(s.dropna().astype(int).tolist()))
            .to_dict()
        )

        users = list(user_test_pos.keys())
        if len(users) == 0:
            return {"AUC": 0.5, "Recall@5": 0.0, "AP": 0.0, "NDCG@5": 0.0}

        if len(users) > max_users:
            users = rng.choice(users, size=max_users, replace=False).tolist()

        user_profiles = self._build_user_profiles(train_reviews)

        all_items = torch.arange(len(self.products_df)).to(self.device)
        recalls, ndcgs, aucs, aps = [], [], [], []

        # ingredient 기반 점수 미리 계산용
        ing_mat = self.ingredient_svd  # (N, svd_dim)

        for u in users:
            pos_items = user_test_pos.get(u, set())
            if len(pos_items) == 0:
                continue

            seen = user_train_seen.get(u, set())
            if u not in user_profiles:
                continue

            uf = user_profiles[int(u)]  # (8,)
            skin_probs = uf[:7]         # (7,)
            user_feature = torch.FloatTensor([uf]).to(self.device)  # [1,8]

            # ✅ 유저의 성분취향 벡터: skin_probs @ skin_pref_matrix
            user_pref_vec = skin_probs @ self.skin_pref_matrix  # (svd_dim,)
            # cosine=dot (둘 다 normalize 된 방향성이라 안정적)
            content_scores = (user_pref_vec @ ing_mat.T).astype(np.float32)  # (N,)

            with torch.no_grad():
                u_rep = user_feature.repeat(len(self.products_df), 1)
                gnn_scores = self.model.score(self.data.x, self.data.edge_index, u_rep, all_items).detach().cpu().numpy()

            # ✅ 하이브리드
            scores = self.alpha * gnn_scores + (1 - self.alpha) * content_scores

            if len(seen) > 0:
                scores[np.array(list(seen), dtype=int)] = -np.inf

            if np.isneginf(scores).all():
                continue

            topk = np.argsort(scores)[-k:][::-1]
            hits = sum([1 for i in topk if i in pos_items])
            recall = hits / min(k, len(pos_items))
            recalls.append(recall)

            rel = np.array([1.0 if i in pos_items else 0.0 for i in topk], dtype=np.float32)
            dcg = np.sum(rel / np.log2(np.arange(2, k + 2)))
            ideal = np.sort(rel)[::-1]
            idcg = np.sum(ideal / np.log2(np.arange(2, k + 2)))
            ndcg = float(dcg / idcg) if idcg > 0 else 0.0
            ndcgs.append(ndcg)

            valid_mask = ~np.isneginf(scores)
            valid_scores = scores[valid_mask]
            valid_items = np.where(valid_mask)[0]
            y_true = np.array([1 if i in pos_items else 0 for i in valid_items], dtype=int)

            if y_true.sum() > 0 and y_true.sum() < len(y_true):
                try:
                    aucs.append(roc_auc_score(y_true, valid_scores))
                except:
                    pass
                try:
                    aps.append(average_precision_score(y_true, valid_scores))
                except:
                    pass

        return {
            "AUC": float(np.mean(aucs)) if len(aucs) else 0.5,
            "Recall@5": float(np.mean(recalls)) if len(recalls) else 0.0,
            "AP": float(np.mean(aps)) if len(aps) else 0.0,
            "NDCG@5": float(np.mean(ndcgs)) if len(ndcgs) else 0.0
        }

    # =========================
    # 추천 (하이브리드)
    # =========================
    def recommend(self, skin_type, category=None, favorite_product_id=None, top_n=5):
        assert hasattr(self, "skin_pref_vec"), "skin_pref_vec가 없음 (train_bpr 먼저 돌려야 함)"

        if category is not None and category != "전체":
            if category not in ALLOWED_CATEGORIES:
                raise ValueError(f"category는 다음 중 하나여야 함: {ALLOWED_CATEGORIES} (또는 None/'전체')")

        self.model.eval()

        # user feature(8): skin one-hot + rating_feat(상수)
        skin_vector = self.skin_type_to_vector.get(skin_type, [1/7] * 7)
        user_feature_1 = torch.FloatTensor([skin_vector + [0.8]]).to(self.device)  # [1,8]

        all_items = torch.arange(len(self.products_df)).to(self.device)

        with torch.no_grad():
            u_rep = user_feature_1.repeat(len(self.products_df), 1)
            gnn_scores = self.model.score(self.data.x, self.data.edge_index, u_rep, all_items).detach().cpu().numpy()

        # ✅ 피부타입 성분취향 점수
        pref = self.skin_pref_vec.get(skin_type, np.zeros(self.svd_dim, dtype=np.float32))
        content_scores = (pref @ self.ingredient_svd.T).astype(np.float32)  # (N,)

        # ✅ 하이브리드
        scores = self.alpha * gnn_scores + (1 - self.alpha) * content_scores

        # 선호 제품 유사도 추가(선택)
        if favorite_product_id and favorite_product_id in self.products_df["product_id"].values:
            fav_encoded = int(self.products_df[self.products_df["product_id"] == favorite_product_id]["product_encoded"].values[0])
            fav_vec = self.ingredient_svd[fav_encoded].reshape(1, -1)
            sim = (fav_vec @ self.ingredient_svd.T).ravel().astype(np.float32)
            scores = 0.7 * scores + 0.3 * sim

        # 카테고리 필터
        if category and category != "전체":
            mask = (self.products_df["category"] == category).values
            if mask.sum() == 0:
                return []
            scores[~mask] = -np.inf
            if np.isneginf(scores).all():
                return []

        top_idx = np.argsort(scores)[-top_n:][::-1]

        recs = []
        for idx in top_idx:
            p = self.products_df.iloc[int(idx)]
            recs.append({
                "product_name": p["product_name"],
                "brand": p["brand"],
                "category": p["category"],
                "rating": float(p.get("product_rating", np.nan)),
                "predicted_score": float(scores[int(idx)]),
                "main_ingredients": p["ingredients_list"][:5],
            })
        return recs

    def print_recommendations(self, recs):
        print("\n" + "=" * 80)
        print("🎁 추천 제품")
        print("=" * 80)
        for i, r in enumerate(recs, 1):
            print(f"\n【 {i}. {r['product_name']} 】")
            print(f"   브랜드: {r['brand']}")
            print(f"   카테고리: {r['category']}")
            if not np.isnan(r["rating"]):
                print(f"   평점: ⭐ {r['rating']:.1f}")
            print(f"   예측 점수: {r['predicted_score']:.3f}")
            print(f"   주요 성분: {', '.join(r['main_ingredients'])}")
            print("-" * 80)

    # =========================
    # 저장/로드
    # =========================
    def save(self, save_dir="singlenode"):
        os.makedirs(save_dir, exist_ok=True)

        torch.save({
            "model_state": self.model.state_dict(),
            "product_encoder": self.product_encoder,
            "category_encoder": self.category_encoder,
            "brand_encoder": self.brand_encoder,
            "user_encoder": self.user_encoder,
            "node_features": self.node_features,
            "edge_index": self.edge_index,
            "svd_dim": self.svd_dim,
            "top_k_edges": self.top_k_edges,
            "alpha": self.alpha,
            "skin_pref_matrix": self.skin_pref_matrix,  # numpy 가능(작음)
        }, os.path.join(save_dir, "single_node_gnn_recommender_bpr.pt"))

        joblib.dump(self.tfidf_vectorizer, os.path.join(save_dir, "tfidf_vectorizer.joblib"))
        joblib.dump(self.svd, os.path.join(save_dir, "svd.joblib"))
        np.save(os.path.join(save_dir, "ingredient_svd.npy"), self.ingredient_svd)

        print("\n✅ 저장 완료")


def main():
    recommender = SingleNodeGNNRecommender(
        products_path="singlenode/final_products.csv",
        reviews_path="singlenode/final_total_reviews.csv",
        svd_dim=100,
        top_k_edges=15
    )

    # ✅ BPR 학습
    recommender.train_bpr(
        epochs=50,
        hidden_channels=128,
        lr=0.001,
        batch_size=1024,
        steps_per_epoch=200,
        pos_threshold=4
    )

    # ✅ 최종 평가
    print("\n" + "=" * 80)
    print("📊 최종 평가 결과")
    print("=" * 80)
    metrics = recommender.evaluate(k=5, max_users=2000)
    print(f"  AUC: {metrics['AUC']:.4f}")
    print(f"  Recall@5: {metrics['Recall@5']:.4f}")
    print(f"  AP: {metrics['AP']:.4f}")
    print(f"  NDCG@5: {metrics['NDCG@5']:.4f}")
    print("=" * 80)

    # ✅ 추천 테스트
    print("\n\n" + "=" * 80)
    print("💡 추천 시스템 테스트")
    print("=" * 80)

    recs = recommender.recommend(
        skin_type="민감성",
        category="에센스/세럼/앰플",
        favorite_product_id=None,
        top_n=5
    )
    recommender.print_recommendations(recs)

    recommender.save("singlenode")


if __name__ == "__main__":
    main()
