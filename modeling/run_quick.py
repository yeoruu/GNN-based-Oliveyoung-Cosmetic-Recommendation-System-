#!/usr/bin/env python
"""
간단 실행 스크립트
빠르게 테스트하고 싶을 때 사용

실행 방법:
python run_quick.py
"""

import os
import sys

print("=" * 80)
print("GNN 추천 시스템 - 빠른 실행".center(80))
print("=" * 80)

# 1. 라이브러리 체크
print("\n🔍 [1] 라이브러리 체크 중...")

required = {
    'torch': 'PyTorch',
    'torch_geometric': 'PyTorch Geometric',
    'sklearn': 'scikit-learn',
    'pandas': 'Pandas',
    'numpy': 'NumPy'
}

missing = []
for module, name in required.items():
    try:
        __import__(module)
        print(f"   ✓ {name}")
    except ImportError:
        print(f"   ✗ {name}")
        missing.append(name)

if missing:
    print(f"\n❌ 누락된 라이브러리: {', '.join(missing)}")
    print("\n설치 명령어:")
    print("   pip install torch torch-geometric scikit-learn pandas numpy")
    sys.exit(1)

print("\n✅ 모든 라이브러리 설치 완료!")

# 2. 데이터 파일 체크
print("\n📁 [2] 데이터 파일 체크 중...")

data_files = [
    'final_products.csv',
    'final_total_reviews.csv'
]

files_ok = True
for file in data_files:
    if os.path.exists(file):
        print(f"   ✓ {file}")
    else:
        print(f"   ✗ {file} - 파일을 찾을 수 없습니다")
        files_ok = False

if not files_ok:
    print("\n⚠️  데이터 파일이 없습니다!")
    print("   gnn_recommender_pytorch.py에서 파일 경로를 수정하거나")
    print("   현재 디렉토리에 CSV 파일을 복사하세요.")
    
    response = input("\n그래도 실행하시겠습니까? (y/n): ")
    if response.lower() != 'y':
        sys.exit(1)

# 3. 실행 옵션 선택
print("\n" + "=" * 80)
print("실행 옵션 선택".center(80))
print("=" * 80)

print("\n1. 빠른 테스트 (10 epochs, 빠름)")
print("2. 표준 학습 (150 epochs, 권장)")
print("3. 사용자 정의")

choice = input("\n선택 (1-3): ").strip()

if choice == '1':
    print("\n🚀 빠른 테스트 모드로 실행...")
    num_epochs = 10
    print(f"   Epochs: {num_epochs}")
elif choice == '2':
    print("\n🚀 표준 모드로 실행...")
    num_epochs = 150
    print(f"   Epochs: {num_epochs}")
elif choice == '3':
    num_epochs = int(input("Epoch 수 입력: "))
    print(f"\n🚀 사용자 정의 모드로 실행...")
    print(f"   Epochs: {num_epochs}")
else:
    print("\n❌ 잘못된 선택입니다.")
    sys.exit(1)

# 4. Config 수정 및 실행
print("\n" + "=" * 80)
print("메인 코드 실행".center(80))
print("=" * 80)

# gnn_recommender_pytorch.py의 Config 수정
import gnn_recommender_pytorch

# Config 오버라이드
gnn_recommender_pytorch.config.NUM_EPOCHS = num_epochs

if choice == '1':
    gnn_recommender_pytorch.config.PATIENCE = 5

print(f"\n설정:")
print(f"   Device: {gnn_recommender_pytorch.config.DEVICE}")
print(f"   Epochs: {gnn_recommender_pytorch.config.NUM_EPOCHS}")
print(f"   Learning Rate: {gnn_recommender_pytorch.config.LEARNING_RATE}")

confirm = input("\n실행하시겠습니까? (y/n): ")
if confirm.lower() != 'y':
    print("취소되었습니다.")
    sys.exit(0)

# 실행
try:
    print("\n\n")
    model, data_loader, rec_system, metrics = gnn_recommender_pytorch.main()
    
    print("\n\n" + "=" * 80)
    print("🎉 실행 완료!".center(80))
    print("=" * 80)
    
    print("\n📊 최종 결과:")
    print(f"   • RMSE: {metrics['RMSE']:.4f}")
    print(f"   • MAE: {metrics['MAE']:.4f}")
    print(f"   • Accuracy: {metrics['Accuracy']:.4f}")
    print(f"   • Hit Rate @10: {metrics['HR@10']:.4f}")
    
    print("\n💾 저장된 파일:")
    print("   • best_gnn_model.pt (최고 성능 모델)")
    
    print("\n💡 사용 방법:")
    print("   recommendations = rec_system.recommend('user_id', top_k=10)")
    
except Exception as e:
    print(f"\n❌ 오류 발생: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
