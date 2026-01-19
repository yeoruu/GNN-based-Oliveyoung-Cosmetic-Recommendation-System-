import os
import re
import time
import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError

INPUT_TABLE2 = "table2_cream_basic.csv"
OUT_TABLE1 = "table1_cream_reviews300.csv"

REVIEWS_PER_PRODUCT = 100
POLITE_SLEEP_SEC = 0.9
SAVE_EVERY_PRODUCTS = 5
RETRY_PER_PRODUCT = 2

# ✅ 필수 피부타입 키워드 정의
REQUIRED_SKIN_TYPES = ["지성", "건성", "복합성", "민감성", "약건성", "트러블성", "중성"]

def clean_text(s):
    """텍스트 정규화"""
    if not s:
        return None
    return re.sub(r"\s+", " ", s).strip() or None

def load_existing_output():
    """이미 저장된 리뷰가 있으면 재시작 시 이어서 수집하기 위해 로드"""
    if not os.path.exists(OUT_TABLE1):
        return pd.DataFrame(columns=[
            "product_id", "product_url", "product_name",
            "product_rating", "user_id", "user_keywords",
            "user_rating", "review_text"
        ])
    
    df = pd.read_csv(OUT_TABLE1)
    # 컬럼 누락 방지
    for c in ["product_id", "product_url", "product_name", "product_rating", "user_id", "user_keywords", "user_rating", "review_text"]:
        if c not in df.columns:
            df[c] = None
    return df

def get_done_map(df_out):
    """product_url별로 이미 수집된 리뷰 개수와 사용자 키워드가 있는 리뷰 개수 반환"""
    if df_out.empty:
        return {}
    
    # 전체 리뷰 개수
    total_map = df_out.groupby("product_url").size().to_dict()
    
    # 사용자 키워드가 있는 리뷰 개수
    if "user_keywords" in df_out.columns:
        keywords_df = df_out[df_out["user_keywords"].notna() & (df_out["user_keywords"].str.strip() != "")]
        keywords_map = keywords_df.groupby("product_url").size().to_dict()
    else:
        keywords_map = {}
    
    # 두 정보를 모두 반환 (dict of dict)
    result = {}
    for url in total_map.keys():
        result[url] = {
            "total": total_map.get(url, 0),
            "with_keywords": keywords_map.get(url, 0)
        }
    return result

def goto_reviews_tab(page, verbose=False):
    """상세페이지에서 '리뷰&셔터' 또는 '리뷰&체험단' 탭 클릭"""
    if verbose:
        print("    🔍 리뷰 탭 찾는 중...")
    
    candidates = [
        'a:has-text("리뷰&셔터")',
        'button:has-text("리뷰&셔터")',
        'text=리뷰&셔터',
        'a:has-text("리뷰&체험단")',
        'button:has-text("리뷰&체험단")',
        'text=리뷰&체험단',
        # 더 일반적인 패턴: "리뷰"가 포함된 탭
        'a:has-text("리뷰")',
        'button:has-text("리뷰")',
    ]
    for sel in candidates:
        loc = page.locator(sel)
        if loc.count() > 0:
            try:
                loc.first.scroll_into_view_if_needed(timeout=2000)
                loc.first.click(timeout=2000)
                page.wait_for_timeout(800)
                if verbose:
                    print(f"    ✅ 리뷰 탭 클릭 성공: {sel}")
                return
            except Exception as e:
                if verbose:
                    print(f"    ⚠️  리뷰 탭 클릭 실패 ({sel}): {str(e)[:50]}")
                continue
    
    # fallback: 리뷰 섹션으로 스크롤 후 다시 시도
    if verbose:
        print("    🔄 스크롤 후 재시도...")
    page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.45);")
    page.wait_for_timeout(600)
    for sel in candidates:
        loc = page.locator(sel)
        if loc.count() > 0:
            try:
                loc.first.click(timeout=2000)
                page.wait_for_timeout(800)
                if verbose:
                    print(f"    ✅ 리뷰 탭 클릭 성공 (재시도): {sel}")
                return
            except Exception as e:
                if verbose:
                    print(f"    ⚠️  리뷰 탭 클릭 실패 ({sel}): {str(e)[:50]}")
                continue
    
    if verbose:
        print("    ❌ 리뷰 탭을 찾을 수 없습니다")
    raise RuntimeError("REVIEWS_TAB_NOT_FOUND")

def set_sort_helpful(page, verbose=False):
    """리뷰 정렬을 '유용한 순'으로 설정"""
    if verbose:
        print("    🔍 유용한 순 정렬 버튼 찾는 중...")
    
    # 정렬 옵션들이 수평으로 나열된 형태에서 '유용한 순' 클릭
    # 여러 방법으로 시도 (버튼, 링크, 텍스트 등)
    candidates = [
        'button:has-text("유용한 순")',
        'a:has-text("유용한 순")',
        'text=유용한 순',
        '*:has-text("유용한 순")',
        'button:has-text("유용한순")',  # 띄어쓰기 없는 버전
        'a:has-text("유용한순")',
    ]
    
    for sel in candidates:
        loc = page.locator(sel)
        if loc.count() > 0:
            try:
                # 이미 선택된 상태인지 확인 (활성화된 스타일이 있는지)
                # 클릭 가능한 상태라면 클릭
                loc.first.scroll_into_view_if_needed(timeout=2000)
                loc.first.click(timeout=2000)
                page.wait_for_timeout(800)  # 정렬 변경 후 리뷰 리스트 갱신 대기
                if verbose:
                    print(f"    ✅ 유용한 순 정렬 성공: {sel}")
                return
            except Exception as e:
                if verbose:
                    print(f"    ⚠️  유용한 순 정렬 실패 ({sel}): {str(e)[:50]}")
                continue
    
    if verbose:
        print("    ⚠️  유용한 순 정렬 버튼을 찾을 수 없음 (계속 진행)")

def extract_product_name(page, verbose=False):
    """상세 상단 제품명 추출"""
    if verbose:
        print("    🔍 제품명 추출 중...")
    
    # UI 버튼 텍스트 제외 목록
    exclude_ui_texts = ["공유하기", "신고하기", "도움", "좋아요", "공감", "추천", "리뷰", "체험단", 
                          "최신순", "유용한 순", "평점 높은 순", "정렬", "더보기", "펼치기", "접기"]
    
    for sel in ["h1", "h2", ".prd_name", "p.prd_name", ".product_name"]:
        loc = page.locator(sel)
        if loc.count() > 0:
            t = clean_text(loc.first.inner_text())
            if t and len(t) >= 2:
                # UI 텍스트 필터링
                if any(exclude in t for exclude in exclude_ui_texts):
                    if verbose:
                        print(f"    ⚠️  UI 텍스트 제외: {t[:50]}...")
                    continue
                # 너무 짧은 텍스트 제외 (2자 이하)
                if len(t) <= 2:
                    continue
                if verbose:
                    print(f"    ✅ 제품명 추출 성공: {t[:50]}...")
                return t
    
    # fallback: 페이지 제목에서 추출 시도
    try:
        title = page.title()
        if title and len(title) > 2:
            # " | " 같은 구분자 제거
            if " | " in title:
                title = title.split(" | ")[0]
            if title and len(title) >= 2 and not any(exclude in title for exclude in exclude_ui_texts):
                if verbose:
                    print(f"    ✅ 제품명 추출 성공 (title): {title[:50]}...")
                return title
    except:
        pass
    
    if verbose:
        print("    ⚠️  제품명 추출 실패")
    return None

def extract_product_rating_in_review_area(page, verbose=False):
    """
    리뷰 영역의 큰 평점(4.7)을 가져오기
    - HTML 구조: <div class="rating-score">4.7</div>
    """
    if verbose:
        print("    🔍 제품 전체 평점 추출 중...")
    
    # 방법 1: rating-score 클래스 직접 찾기
    try:
        rating_loc = page.locator(".rating-score").first
        if rating_loc.count() > 0:
            rating_text = clean_text(rating_loc.inner_text())
            if rating_text:
                m = re.search(r"([0-5]\.\d)", rating_text)
                if m:
                    rating = m.group(1)
                    if verbose:
                        print(f"    ✅ 제품 전체 평점 추출 성공 (.rating-score): {rating}")
                    return rating
    except Exception as e:
        if verbose:
            print(f"    ⚠️  평점 추출 실패 (.rating-score): {str(e)[:50]}")
        pass
    
    # 방법 2: star-container 안의 rating-score
    try:
        container = page.locator(".star-container .rating-score").first
        if container.count() > 0:
            rating_text = clean_text(container.inner_text())
            if rating_text:
                m = re.search(r"([0-5]\.\d)", rating_text)
                if m:
                    rating = m.group(1)
                    if verbose:
                        print(f"    ✅ 제품 전체 평점 추출 성공 (.star-container .rating-score): {rating}")
                    return rating
    except Exception as e:
        if verbose:
            print(f"    ⚠️  평점 추출 실패 (star-container): {str(e)[:50]}")
        pass
    
    # 방법 3: '총' 텍스트를 anchor로 잡기 (기존 방법)
    try:
        anchor = page.locator("text=총").first
        if anchor.count() > 0:
            box = anchor.locator("xpath=ancestor::*[1]")
            t = box.inner_text()
            m = re.search(r"([0-5]\.\d)", t)
            if m:
                rating = m.group(1)
                if verbose:
                    print(f"    ✅ 제품 전체 평점 추출 성공 (총 앵커): {rating}")
                return rating
    except Exception as e:
        if verbose:
            print(f"    ⚠️  평점 추출 실패 (총 앵커): {str(e)[:50]}")
        pass
    
    # fallback: 화면 HTML에서 0~5.x 하나 찾기(정확도 낮지만 백업)
    try:
        html = page.content()
        m = re.search(r"rating-score[^>]*>([0-5]\.\d)", html)
        if m:
            rating = m.group(1)
            if verbose:
                print(f"    ⚠️  제품 전체 평점 추출 (HTML fallback): {rating}")
            return rating
    except:
        pass
    
    if verbose:
        print("    ❌ 제품 전체 평점 추출 실패")
    return None

def get_review_cards(page, verbose=False):
    """리뷰 카드 locator를 최대한 robust하게 잡기"""
    # HTML 구조: <li> > <oy-review-review-item> > <div class="review-item"> > <div class="inner">
    selectors = [
        # 방법 1: oy-review-review-item (Shadow DOM 포함)
        ('oy-review-review-item', "oy-review-review-item 컴포넌트"),
        # 방법 2: div.review-item (review-item 클래스)
        ('div.review-item', "div.review-item 클래스"),
        # 방법 3: div.inner (실제 구조 - 리뷰 카드 내부)
        ('div.inner', "div.inner 클래스"),
        # 방법 4: oy-review-review-content가 있는 div
        ('div:has(oy-review-review-content)', "oy-review-review-content 포함 div"),
        # 방법 5: div.name이 있는 div (닉네임이 있는 카드)
        ('div:has(div.name)', "div.name 포함 div"),
        # 방법 6: div.rating이 있는 div (별점이 있는 카드)
        ('div:has(div.rating)', "div.rating 포함 div"),
        # 방법 7: '신고하기' 텍스트가 있는 카드의 부모
        ('xpath=//button[contains(text(),"신고하기")]/ancestor::div[@class="review-item" or @class="inner"][1]', "신고하기 앵커"),
        # 방법 8: 리뷰 관련 클래스
        ('.review_list li', "review_list li"),
        ('li.review', "li.review"),
        # 방법 9: 일반 li (최후 수단)
        ('li', "일반 li"),
    ]
    
    for selector, desc in selectors:
        try:
            loc = page.locator(selector)
            count = loc.count()
            if count > 0:
                if verbose:
                    print(f"      ✅ 리뷰 카드 발견 ({desc}): {count}개")
                return loc
        except:
            continue
    
    if verbose:
        print("      ⚠️  리뷰 카드를 찾을 수 없음")
    # 최후 fallback
    return page.locator("div")

def extract_user_rating_from_card(card):
    """
    리뷰 카드 내 사용자 별점 추출 (Shadow DOM: oy-review-star-icon)
    - 원리: rating div 안에 oy-review-star-icon 5개가 있고,
            채워진 별은 shadowRoot 내부 path의 fill="#FF5753"로 표시됨.
    """
    # 1) rating 컨테이너부터 잡기 (예시: <div class="rating">)
    rating = card.locator("div.rating").first
    if rating.count() == 0:
        # fallback: 별 컴포넌트가 있는지로 직접 찾기
        rating = card
    
    # 2) 별 컴포넌트들(보통 5개)
    stars = rating.locator("oy-review-star-icon")
    n = stars.count()
    if n == 0:
        return None
    
    filled = 0
    for i in range(n):
        star = stars.nth(i)
        
        # ✅ shadowRoot 내부 svg path의 fill 속성 확인
        # 채워진 별: fill="#FF5753"
        # 빈 별: fill="none"
        try:
            path = star.locator("svg path").first
            fill = path.get_attribute("fill")  # "#FF5753" 또는 "none"
            if fill and fill.lower() == "#ff5753":
                filled += 1
        except:
            pass
    
    # 별 개수는 1~5로 반환
    if 0 <= filled <= 5:
        return filled
    
    # fallback: 혹시 fill 횟수 계산이 꼬이면 5로 클램프
    return max(0, min(5, filled))

def parse_review_card(card):
    """
    카드에서:
      - user_id: 마스킹 닉네임 (예: wish****)
      - user_keywords: 닉네임 아래 '복합성·...' 같은 라인(없을 수 있음)
      - user_rating: 사용자별 별점
      - review_text: 본문
    """
    # 닉네임: HTML 구조 <div class="name">wish****</div> 또는 <div class="name">뎡밍</div>
    # UI 버튼 텍스트 제외 목록
    exclude_ui_texts = ["신고하기", "도움", "좋아요", "공감", "추천", "공유하기", "리뷰", "체험단", 
                        "최신순", "유용한 순", "평점 높은 순", "정렬", "더보기", "펼치기", "접기",
                        "매장", "온라인", "구매", "후기", "평점", "한달이상사용"]
    
    user_id = None
    try:
        # 방법 1: div.name 직접 찾기 (Shadow DOM 자동 처리)
        name_loc = card.locator("div.name").first
        if name_loc.count() > 0:
            t = clean_text(name_loc.inner_text())
            if t and len(t) <= 20 and len(t) >= 1:
                # UI 텍스트 필터링
                if not any(exclude in t for exclude in exclude_ui_texts):
                    # 날짜 형식 제외
                    if not re.search(r"\d{4}\.\d{2}\.\d{2}", t):
                        user_id = t
    except:
        pass
    
    # 방법 2: div.name-wrap > div.name 찾기
    if not user_id:
        try:
            name_wrap = card.locator("div.name-wrap div.name").first
            if name_wrap.count() > 0:
                t = clean_text(name_wrap.inner_text())
                if t and len(t) <= 20 and len(t) >= 1:
                    if not any(exclude in t for exclude in exclude_ui_texts):
                        if not re.search(r"\d{4}\.\d{2}\.\d{2}", t):
                            user_id = t
        except:
            pass
    
    # 방법 3: fallback - 일반 노드에서 찾기
    if not user_id:
        nodes = card.locator("strong, b, span, p, div")
        for i in range(min(nodes.count(), 30)):
            t = clean_text(nodes.nth(i).inner_text())
            if not t:
                continue
            # UI 텍스트 필터링
            if any(exclude in t for exclude in exclude_ui_texts):
                continue
            # 날짜 형식 제외
            if re.search(r"\d{4}\.\d{2}\.\d{2}", t):
                continue
            # 닉네임 패턴: 한글/영문/숫자/언더스코어 + 별표 마스킹 (별표는 선택적)
            # 한글도 포함 (예: "뎡밍")
            if re.match(r"^[가-힣a-zA-Z0-9_]{1,15}(\*{0,8})?$", t) and len(t) <= 20 and len(t) >= 1:
                # "·"가 있으면 키워드일 가능성이 높음
                if "·" in t:
                    continue
                user_id = t
                break
    
    # 사용자 키워드: HTML 구조 <div class="skin-types"> 안에 여러 <span class="skin-type"> 요소들
    # 예: "지성", "겨울쿨톤", "잡티", "모공" 등
    user_keywords = None
    keywords_list = []
    
    # 방법 1: div.skin-types 안의 span.skin-type 직접 찾기
    try:
        skin_types_div = card.locator("div.skin-types").first
        if skin_types_div.count() > 0:
            skin_type_spans = skin_types_div.locator("span.skin-type")
            for i in range(skin_type_spans.count()):
                t = clean_text(skin_type_spans.nth(i).inner_text())
                if t and len(t) <= 30:  # 키워드는 보통 짧음
                    # 날짜나 매장 정보 제외
                    if not re.search(r"\d{4}\.\d{2}\.\d{2}", t) and t != user_id:
                        keywords_list.append(t)
            
            if keywords_list:
                # 여러 키워드를 " | "로 연결 (예: "지성 | 겨울쿨톤 | 잡티 | 모공")
                user_keywords = " | ".join(keywords_list)
    except:
        pass
    
    # 방법 2: fallback - div.info 내부에서 찾기
    if not user_keywords:
        try:
            info_nodes = card.locator("div.info span, div.info div, div.name-wrap span, div.name-wrap div")
            for i in range(min(info_nodes.count(), 20)):
                t = clean_text(info_nodes.nth(i).inner_text())
                if not t or len(t) > 80:
                    continue
                # "·" 구분자가 있거나 피부타입/톤 관련 키워드가 포함된 경우
                if ("·" in t) or any(k in t for k in ["지성", "건성", "복합성", "민감", "트러블", "여드름", "각질", "모공", 
                                                        "쿨톤", "웜톤", "탄력", "주름", "미백", "톤업", "잡티"]):
                    # 날짜/매장 같은 정보 줄(2025.12.08 등) 제외
                    if re.search(r"\d{4}\.\d{2}\.\d{2}", t):
                        continue
                    # 닉네임과 동일한 텍스트 제외
                    if t == user_id:
                        continue
                    user_keywords = t
                    break
        except:
            pass
    
    # 사용자별 별점
    user_rating = extract_user_rating_from_card(card)
    
    # 리뷰 본문: HTML 구조 
    # <oy-review-review-content> > <div class="review-content-container"> > <div class="content"> > <p>
    review_text = None
    
    # 시스템 메시지 제외 목록
    exclude_system_texts = [
        "해당 리뷰는 성분과 내용물이 동일", "성분과 내용물이 동일", 
        "동일한 제품", "중복 리뷰", "리뷰가 삭제되었습니다"
    ]
    
    # 방법 1: oy-review-review-content > .content > p 직접 찾기 (가장 정확)
    try:
        content_loc = card.locator("oy-review-review-content .content p").first
        if content_loc.count() > 0:
            t = clean_text(content_loc.inner_text())
            if t and len(t) >= 10:  # 최소 길이를 10자로 낮춤 (짧은 리뷰도 포함)
                # 시스템 메시지 필터링
                if not any(exclude in t for exclude in exclude_system_texts):
                    review_text = t
    except:
        pass
    
    # 방법 2: .review-content-container .content p
    if not review_text:
        try:
            content_loc = card.locator(".review-content-container .content p").first
            if content_loc.count() > 0:
                t = clean_text(content_loc.inner_text())
                if t and len(t) >= 10:
                    # 시스템 메시지 필터링
                    if not any(exclude in t for exclude in exclude_system_texts):
                        review_text = t
        except:
            pass
    
    # 방법 3: oy-review-review-content 내부의 p 태그 직접 찾기
    if not review_text:
        try:
            content_loc = card.locator("oy-review-review-content p").first
            if content_loc.count() > 0:
                t = clean_text(content_loc.inner_text())
                if t and len(t) >= 10:
                    # 시스템 메시지 필터링
                    if not any(exclude in t for exclude in exclude_system_texts):
                        review_text = t
        except:
            pass
    
    # 방법 4: .content p 찾기
    if not review_text:
        try:
            content_loc = card.locator(".content p").first
            if content_loc.count() > 0:
                t = clean_text(content_loc.inner_text())
                if t and len(t) >= 10:
                    # 시스템 메시지 필터링
                    if not any(exclude in t for exclude in exclude_system_texts):
                        review_text = t
        except:
            pass
    
    # 방법 5: oy-review-review-content 전체 텍스트
    if not review_text:
        try:
            content_loc = card.locator("oy-review-review-content").first
            if content_loc.count() > 0:
                t = clean_text(content_loc.inner_text())
                if t and len(t) >= 10:
                    # 시스템 메시지 필터링
                    if not any(exclude in t for exclude in exclude_system_texts):
                        review_text = t
        except:
            pass
    
    # 방법 6: fallback - 일반적인 텍스트 노드에서 긴 텍스트 찾기
    if not review_text:
        best = ""
        best_len = 0
        nodes = card.locator("strong, b, span, p, div")
        
        # 시스템 메시지 제외 목록
        exclude_system_texts = [
            "해당 리뷰는 성분과 내용물이 동일", "성분과 내용물이 동일", 
            "동일한 제품", "중복 리뷰", "리뷰가 삭제되었습니다",
            "신고하기", "도움", "좋아요", "공감", "추천", "매장", "온라인",
            "공유하기", "리뷰", "체험단", "최신순", "유용한 순", "평점 높은 순"
        ]
        
        for i in range(min(nodes.count(), 80)):
            t = clean_text(nodes.nth(i).inner_text())
            if not t:
                continue
            # UI 요소 및 시스템 메시지 제외
            if any(exclude in t for exclude in exclude_system_texts):
                continue
            # 날짜 형식 제외
            if re.search(r"\d{4}\.\d{2}\.\d{2}", t):
                continue
            # 너무 짧거나 너무 긴 텍스트 제외 (리뷰는 보통 10자 이상, 500자 이하)
            if len(t) >= 10 and len(t) <= 500 and len(t) > best_len:
                # 닉네임이나 키워드와 동일한 텍스트 제외
                if t != user_id and t != user_keywords:
                    # 키워드 리스트에 포함된 텍스트도 제외
                    if not any(kw in t for kw in (keywords_list if keywords_list else [])):
                        best = t
                        best_len = len(t)
        review_text = best if best else None
    
    return user_id, user_keywords, user_rating, review_text

def click_more_if_exists(page, verbose=False) -> bool:
    """리뷰 목록에서 '더보기' 버튼이 있으면 클릭하고 새로운 리뷰가 로드될 때까지 대기"""
    # 클릭 전 리뷰 카드 개수 확인
    before_count = page.locator("oy-review-review-item").count()
    
    for sel in ['button:has-text("더보기")', 'a:has-text("더보기")', 'text=더보기']:
        loc = page.locator(sel)
        if loc.count() > 0:
            try:
                loc.first.scroll_into_view_if_needed(timeout=2000)
                loc.first.click(timeout=2000)
                
                # 새로운 리뷰가 로드될 때까지 대기 (최대 5초)
                if verbose:
                    print(f"    ⏳ '더보기' 클릭 후 새 리뷰 로딩 대기 중... (현재: {before_count}개)")
                
                # 스크롤을 먼저 해서 lazy loading 유도
                page.mouse.wheel(0, 1000)
                page.wait_for_timeout(300)
                
                for wait_attempt in range(25):  # 최대 5초 대기 (200ms * 25)
                    page.wait_for_timeout(200)
                    after_count = page.locator("oy-review-review-item").count()
                    if after_count > before_count:
                        if verbose:
                            print(f"    ✅ 새로운 리뷰 로드됨: {before_count}개 → {after_count}개 (+{after_count - before_count}개)")
                        # 추가로 스크롤하여 더 많은 리뷰 로드 유도
                        page.mouse.wheel(0, 1500)
                        page.wait_for_timeout(500)
                        return True
                
                # 리뷰 개수가 변하지 않아도 추가 대기 및 스크롤 (lazy loading 대비)
                page.mouse.wheel(0, 2000)
                page.wait_for_timeout(800)
                after_count = page.locator("oy-review-review-item").count()
                if after_count > before_count:
                    if verbose:
                        print(f"    ✅ 새로운 리뷰 로드됨 (지연): {before_count}개 → {after_count}개 (+{after_count - before_count}개)")
                    return True
                elif verbose:
                    print(f"    ⚠️  리뷰 개수 변화 없음: {before_count}개 → {after_count}개")
                
                if verbose:
                    print("    ✅ '더보기' 버튼 클릭 성공 (리뷰 개수 변화 없음)")
                return True
            except Exception as e:
                if verbose:
                    print(f"    ⚠️  '더보기' 버튼 클릭 실패: {str(e)[:50]}")
                continue
    if verbose:
        print("    ⚠️  '더보기' 버튼 없음")
    return False

def collect_helpful_reviews_for_product(page, product_url, limit=None, verbose=False):
    """제품 페이지에서 유용한 순 리뷰 수집"""
    start_time = time.time()  # 전체 수집 시작 시간 (소요 시간 측정용)
    
    if verbose:
        print("    🌐 페이지 로딩 중...")
    page.goto(product_url, wait_until="domcontentloaded")
    page.wait_for_timeout(900)
    if verbose:
        print("    ✅ 페이지 로딩 완료")
    
    product_name = extract_product_name(page, verbose=verbose)
    
    goto_reviews_tab(page, verbose=verbose)
    
    # 리뷰 탭으로 이동한 후 리뷰가 로드될 때까지 대기
    if verbose:
        print("    ⏳ 리뷰 리스트 로딩 대기 중...")
    for wait_attempt in range(10):
        page.wait_for_timeout(500)
        # 리뷰 카드가 나타나는지 확인
        test_cards = get_review_cards(page, verbose=False)
        if test_cards.count() > 0:
            if verbose:
                print(f"    ✅ 리뷰 리스트 로드 완료 ({test_cards.count()}개 카드 발견)")
            break
        if verbose and wait_attempt == 4:
            print(f"    ⏳ 리뷰 로딩 대기 중... ({wait_attempt+1}/10)")
    else:
        if verbose:
            print("    ⚠️  리뷰 카드가 로드되지 않음 (계속 진행)")
    
    # ✅ 유용한 순으로 정렬
    set_sort_helpful(page, verbose=verbose)
    
    # 정렬 후 다시 대기
    if verbose:
        print("    ⏳ 정렬 후 리뷰 리스트 갱신 대기 중...")
    page.wait_for_timeout(1000)
    
    # 전체 평점은 리뷰 탭 진입 후 추출하는 게 가장 안전
    product_rating = extract_product_rating_in_review_area(page, verbose=verbose)
    
    if verbose:
        print(f"    📋 리뷰 수집 시작... (목표: 피부타입 있는 리뷰 {limit if limit else 100}개, 정렬: 유용한 순)")
        print(f"    🎯 필수 피부타입: {', '.join(REQUIRED_SKIN_TYPES)}")
    collected = []
    seen = set()
    guard = 0
    iteration = 0
    max_iterations = 200  # 최대 반복 횟수 제한
    target_count = limit if limit else 100  # 목표 리뷰 개수
    
    while len(collected) < target_count and guard < 30 and iteration < max_iterations:
        iteration += 1
        cards = get_review_cards(page, verbose=verbose)
        n = min(cards.count(), 250)
        
        if n == 0 and verbose:
            print(f"    ⚠️  반복 {iteration}: 리뷰 카드가 0개입니다. 페이지 구조 확인 필요")

        if verbose:
            print(f"    🔄 반복 {iteration}: 발견된 리뷰 카드 {n}개, 현재 수집: {len(collected)}개 (목표: {target_count}개)")
        
        before = len(collected)
        parsed_count = 0
        skipped_no_text = 0
        skipped_no_keywords = 0  # ✅ 키워드 자체가 없는 경우
        skipped_no_skin_type = 0  # ✅ 키워드는 있지만 필수 피부타입이 없는 경우
        skipped_duplicate = 0
        
        for i in range(n):
            if len(collected) >= target_count:
                break
            card = cards.nth(i)
            
            try:
                user_id, user_keywords, user_rating, review_text = parse_review_card(card)
            except Exception as e:
                if verbose and i < 3:  # 처음 몇 개만 로그
                    print(f"      ⚠️  카드 {i+1} 파싱 실패: {str(e)[:50]}")
                continue
            
            # 리뷰 본문 체크
            if not review_text:
                skipped_no_text += 1
                continue
            
            # ✅ 사용자 키워드 체크
            if not user_keywords or (isinstance(user_keywords, str) and user_keywords.strip() == ""):
                skipped_no_keywords += 1
                if verbose and skipped_no_keywords <= 3:
                    print(f"      ⚠️  카드 {i+1}: 사용자 키워드 없음 (user_id: {user_id})")
                continue
            
            # ✅ 필수 피부타입 키워드 체크 (지성, 건성, 복합성, 민감성, 약건성, 트러블성, 중성)
            has_skin_type = any(skin_type in user_keywords for skin_type in REQUIRED_SKIN_TYPES)
            if not has_skin_type:
                skipped_no_skin_type += 1
                if verbose and skipped_no_skin_type <= 3:
                    print(f"      ⚠️  카드 {i+1}: 필수 피부타입 없음 (키워드: {user_keywords[:50]})")
                continue
            
            key = (user_id or "") + "::" + review_text[:80]
            if key in seen:
                skipped_duplicate += 1
                continue
            seen.add(key)
            parsed_count += 1
            
            collected.append({
                "product_name": product_name,
                "product_rating": product_rating,  # 전체 평점(모든 리뷰 행에서 동일)
                "user_id": user_id,
                "user_keywords": user_keywords,
                "user_rating": user_rating,  # 사용자별 평점(리뷰마다 다름)
                "review_text": review_text
            })
        
        if verbose:
            print(f"      └─ 새로 추가: {parsed_count}개, 텍스트 없음: {skipped_no_text}개, " +
                  f"키워드 없음: {skipped_no_keywords}개, 피부타입 없음: {skipped_no_skin_type}개, " +
                  f"중복: {skipped_duplicate}개")
        
        # 더보기/스크롤로 추가 로딩
        if len(collected) == before:
            if verbose:
                print("    🔄 새로운 리뷰 없음, 더보기/스크롤 시도...")
            
            # 현재 리뷰 카드 개수 확인
            current_card_count = cards.count()
            card_count_changed = False
            
            # 더보기 버튼 클릭 시도
            progressed = click_more_if_exists(page, verbose=verbose)
            
            # 더보기 클릭 후 리뷰 카드 개수 다시 확인
            if progressed:
                page.wait_for_timeout(500)  # 추가 대기
                new_cards = get_review_cards(page, verbose=False)
                new_card_count = new_cards.count()
                if verbose:
                    print(f"    📊 리뷰 카드 개수: {current_card_count}개 → {new_card_count}개")
                if new_card_count > current_card_count:
                    card_count_changed = True
            
            # 더보기가 없거나 효과가 없으면 스크롤
            if not progressed or not card_count_changed:
                if verbose:
                    print("    📜 스크롤 다운 (lazy loading 유도)...")
                # 여러 번 스크롤하여 lazy loading 유도 (더 적극적으로)
                for scroll_i in range(5):  # 3회 -> 5회로 증가
                    page.mouse.wheel(0, 2500)  # 스크롤 거리도 증가
                    page.wait_for_timeout(500)  # 대기 시간도 증가
                page.wait_for_timeout(1200)  # 스크롤 후 추가 대기 시간 증가
                
                # 스크롤 후 리뷰 카드 개수 다시 확인
                new_cards_after_scroll = get_review_cards(page, verbose=False)
                new_count_after_scroll = new_cards_after_scroll.count()
                if verbose:
                    print(f"    📊 스크롤 후 리뷰 카드 개수: {new_count_after_scroll}개 (이전: {current_card_count}개)")
                if new_count_after_scroll > current_card_count:
                    card_count_changed = True
            
            # ✅ 스크롤/더보기 후에도 카드 개수가 변화하지 않았을 때만 guard 증가
            if not card_count_changed:
                guard += 1
                if verbose:
                    print(f"    ⚠️  카드 개수 변화 없음 → guard 증가 ({guard}/30회)")
                if verbose and guard >= 3 and guard % 5 == 0:  # 5회마다 로그 출력
                    print(f"    ⚠️  진행 없음 ({guard}/30회) - 현재 수집: {len(collected)}개 / {target_count}개 (목표)")
            else:
                # 카드 개수가 증가했지만 실제로 새로운 리뷰를 수집하지 못했다면 guard 증가
                # (카드 개수만 증가하고 파싱할 수 없는 경우를 대비)
                if len(collected) == before:
                    guard += 1
                    if verbose:
                        print(f"    ⚠️  카드 개수는 증가했지만 새로운 리뷰 수집 실패 → guard 증가 ({guard}/30회)")
                else:
                    # 실제로 새로운 리뷰를 수집했으면 guard 리셋
                    guard = 0
                    if verbose:
                        print(f"    ✅ 새로운 리뷰 수집 성공 → guard 리셋")
        else:
            guard = 0
            if verbose:
                print(f"    ✅ 진행 중... 현재 수집: {len(collected)}개 / {target_count}개 (목표)")
    
    # 종료 이유 확인
    elapsed_total = time.time() - start_time
    if len(collected) >= target_count:
        if verbose:
            print(f"    ✅ 리뷰 수집 완료: {len(collected)}개 (목표: {target_count}개, 소요 시간: {elapsed_total:.1f}초)")
    elif guard >= 30:
        if verbose:
            print(f"    ⚠️  종료: guard 한계 도달 ({guard}/30회, 소요 시간: {elapsed_total:.1f}초) - 수집된 리뷰: {len(collected)}개")
    elif iteration >= max_iterations:
        if verbose:
            print(f"    ⚠️  종료: 최대 반복 횟수 도달 ({iteration}/{max_iterations}회, 소요 시간: {elapsed_total:.1f}초) - 수집된 리뷰: {len(collected)}개")
    else:
        if verbose:
            print(f"    ✅ 리뷰 수집 완료: {len(collected)}개 (목표: {target_count}개, 소요 시간: {elapsed_total:.1f}초)")
    
    return product_name, product_rating, collected

def test_single_product(product_url=None, headless=False, verbose=True, limit=120):
    """
    단일 제품으로 리뷰 크롤링 테스트
    product_url이 None이면 CSV의 첫 번째 제품 사용
    """
    if product_url is None:
        if not os.path.exists(INPUT_TABLE2):
            print(f"❌ 오류: {INPUT_TABLE2} 파일을 찾을 수 없습니다.")
            return None
        t2 = pd.read_csv(INPUT_TABLE2)[["product_id", "product_url"]].dropna().drop_duplicates("product_url").reset_index(drop=True)
        if len(t2) == 0:
            print("❌ 오류: CSV 파일에 제품이 없습니다.")
            return None
        product_url = t2.iloc[0]["product_url"]
        product_id = str(t2.iloc[0]["product_id"])
        print(f"📋 CSV의 첫 번째 제품 사용: {product_id}")
    else:
        product_id = "TEST"
    
    print("="*60)
    print("🧪 단일 제품 테스트 모드")
    print("="*60)
    print(f"제품 URL: {product_url}")
    print(f"목표 리뷰 수: {limit}개 (피부타입 필수)")
    print(f"필수 피부타입: {', '.join(REQUIRED_SKIN_TYPES)}")
    print(f"정렬 방식: 유용한 순")
    print(f"Headless 모드: {headless}")
    print("="*60)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            locale="ko-KR",
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
            viewport={"width": 1400, "height": 900}
        )
        page = context.new_page()
        
        try:
            product_name, product_rating, reviews = collect_helpful_reviews_for_product(
                page, product_url, limit=limit, verbose=verbose
            )
            
            # 결과 출력
            print("\n" + "="*60)
            print("✅ 테스트 완료!")
            print("="*60)
            print(f"제품명: {product_name}")
            print(f"제품 전체 평점: {product_rating}")
            print(f"수집된 리뷰 수: {len(reviews)}개 (피부타입 필수, 유용한 순)")
            print("="*60)
            
            if reviews:
                # DataFrame 생성
                df = pd.DataFrame([{
                    "product_id": product_id,
                    "product_url": product_url,
                    "product_name": r["product_name"],
                    "product_rating": r["product_rating"],
                    "user_id": r["user_id"],
                    "user_keywords": r["user_keywords"],
                    "user_rating": r["user_rating"],
                    "review_text": r["review_text"],
                } for r in reviews])
                
                print("\n📋 수집된 리뷰 미리보기 (최대 10개):")
                print("-"*60)
                print(df[["user_id", "user_rating", "user_keywords", "review_text"]].head(10).to_string())
                print("-"*60)
                
                # 통계 정보
                if "user_rating" in df.columns and df["user_rating"].notna().any():
                    print(f"\n📊 사용자 평점 통계:")
                    print(f"  평균: {df['user_rating'].mean():.2f}")
                    print(f"  최고: {df['user_rating'].max()}")
                    print(f"  최저: {df['user_rating'].min()}")
                    print(f"  분포: {df['user_rating'].value_counts().sort_index().to_dict()}")
                
                # ✅ 피부타입 통계
                if "user_keywords" in df.columns:
                    print(f"\n📊 피부타입 분포:")
                    skin_type_counts = {}
                    for keywords in df["user_keywords"]:
                        if keywords:
                            for skin_type in REQUIRED_SKIN_TYPES:
                                if skin_type in keywords:
                                    skin_type_counts[skin_type] = skin_type_counts.get(skin_type, 0) + 1
                    for skin_type, count in sorted(skin_type_counts.items(), key=lambda x: -x[1]):
                        print(f"  {skin_type}: {count}개")
                
                return df
            else:
                print("\n⚠️  수집된 리뷰가 없습니다.")
                return None
                
        except Exception as e:
            print(f"\n❌ 오류 발생: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            browser.close()

def scrape_all_products_reviews(headless=True, verbose=True, start_idx=0, end_idx=None):
    """
    모든 제품의 리뷰 크롤링 (목표: 각 제품당 피부타입이 있는 리뷰 100개, 유용한 순)
    
    Args:
        headless: 브라우저를 숨김 모드로 실행할지 여부
        verbose: 상세 로그 출력 여부
        start_idx: 시작 인덱스 (BASIC CSV의 몇 번째 제품부터 시작할지, 0부터 시작)
        end_idx: 끝 인덱스 (BASIC CSV의 몇 번째 제품까지 할지, None이면 끝까지)
    """
    if verbose:
        print("="*60)
        print("💬 올리브영 리뷰 크롤러 시작 (에센스/세럼/앰플)")
        print("="*60)
        print(f"입력 파일: {INPUT_TABLE2}")
        print(f"출력 파일: {OUT_TABLE1}")
        print(f"목표: 각 제품당 피부타입 있는 리뷰 {REVIEWS_PER_PRODUCT}개")
        print(f"필수 피부타입: {', '.join(REQUIRED_SKIN_TYPES)}")
        print(f"정렬 방식: 유용한 순")
        if start_idx > 0 or end_idx is not None:
            print(f"범위: 인덱스 {start_idx}부터 {end_idx if end_idx is not None else '끝까지'}")
        print("="*60)
    
    if not os.path.exists(INPUT_TABLE2):
        print(f"❌ 오류: {INPUT_TABLE2} 파일을 찾을 수 없습니다.")
        return None
    
    t2 = pd.read_csv(INPUT_TABLE2)[["product_id", "product_url"]].dropna().drop_duplicates("product_url").reset_index(drop=True)
    
    # 인덱스 범위 지정
    total = len(t2)
    if start_idx < 0:
        start_idx = 0
    if start_idx >= total:
        print(f"❌ 오류: 시작 인덱스({start_idx})가 총 제품 수({total})보다 큽니다.")
        return None
    if end_idx is None:
        end_idx = total
    if end_idx > total:
        end_idx = total
    if start_idx >= end_idx:
        print(f"❌ 오류: 시작 인덱스({start_idx})가 끝 인덱스({end_idx})보다 크거나 같습니다.")
        return None
    
    # 지정된 범위만 사용
    t2 = t2.iloc[start_idx:end_idx].reset_index(drop=True)
    range_total = len(t2)
    
    out_df = load_existing_output()
    done_map = get_done_map(out_df)
    
    # 전체 리뷰가 REVIEWS_PER_PRODUCT개 이상인 제품 수
    completed = len([url for url, info in done_map.items() 
                     if (isinstance(info, dict) and info.get("total", 0) >= REVIEWS_PER_PRODUCT) or
                        (isinstance(info, (int, float)) and info >= REVIEWS_PER_PRODUCT)])
    remaining = range_total - completed
    
    if verbose:
        print(f"\n📊 BASIC CSV 총 제품 수: {total}개")
        print(f"📊 크롤링 범위: 인덱스 {start_idx}~{end_idx-1} ({range_total}개 제품)")
        print(f"📊 이미 완료된 제품 (리뷰 {REVIEWS_PER_PRODUCT}개 이상): {completed}개")
        print(f"📊 남은 제품: {remaining}개")
        print("="*60)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            locale="ko-KR",
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
            viewport={"width": 1400, "height": 900}
        )
        page = context.new_page()
        
        processed = 0
        total_reviews = 0
        last_saved_idx = start_idx - 1  # 마지막으로 저장된 인덱스 추적
        
        for idx, (_, row) in enumerate(t2.iterrows(), 0):
            # 실제 BASIC CSV의 인덱스 (start_idx부터 시작)
            actual_idx = start_idx + idx
            pid = str(row["product_id"])
            url = str(row["product_url"])
            
            # 완료 여부 확인: 전체 리뷰가 REVIEWS_PER_PRODUCT개 이상인지
            done_info = done_map.get(url, {})
            total_count = done_info.get("total", 0) if isinstance(done_info, dict) else (done_info if isinstance(done_info, (int, float)) else 0)
            
            if total_count >= REVIEWS_PER_PRODUCT:
                if verbose:
                    print(f"[인덱스 {actual_idx} / 범위 {start_idx}~{end_idx-1}] ⏭️  건너뜀 (이미 완료, {total_count}개 리뷰): {url}")
                continue
            
            ok = False
            err = ""
            product_name = None
            product_rating = None
            reviews = []
            
            if verbose:
                print(f"\n[인덱스 {actual_idx} / 범위 {start_idx}~{end_idx-1}] 🔍 처리 중: {url}")
            
            for attempt in range(RETRY_PER_PRODUCT + 1):
                try:
                    product_name, product_rating, reviews = collect_helpful_reviews_for_product(
                        page, url, limit=REVIEWS_PER_PRODUCT, verbose=verbose
                    )
                    ok = True
                    err = ""
                    if verbose:
                        print(f"  ✅ 전체 성공! (시도 {attempt+1}/{RETRY_PER_PRODUCT+1})")
                        print(f"  📝 최종 수집된 리뷰: {len(reviews)}개 (피부타입 필수, 유용한 순)")
                        print(f"  ⭐ 제품 전체 평점: {product_rating}")
                    break
                except Exception as e:
                    err = f"{type(e).__name__}:{str(e)}"
                    if verbose:
                        if attempt < RETRY_PER_PRODUCT:
                            print(f"  ⚠️  실패 (시도 {attempt+1}/{RETRY_PER_PRODUCT+1}): {err}")
                            print(f"  🔄 재시도 중...")
                        else:
                            print(f"  ❌ 최종 실패: {err}")
                    if attempt < RETRY_PER_PRODUCT:
                        page.wait_for_timeout(1200)
            
            if reviews:
                block = pd.DataFrame([{
                    "product_id": pid,
                    "product_url": url,
                    "product_name": r["product_name"],
                    "product_rating": r["product_rating"],  # ✅ 제품 전체 평점(모든 행 동일)
                    "user_id": r["user_id"],
                    "user_keywords": r["user_keywords"],
                    "user_rating": r["user_rating"],  # ✅ 리뷰 작성자 별점(행마다 다름)
                    "review_text": r["review_text"],
                } for r in reviews])
                
                # 이미 일부 저장된 제품이면 중복 제거를 위해 concat 후 drop
                out_df = pd.concat([out_df, block], ignore_index=True)
                out_df = out_df.drop_duplicates(
                    subset=["product_url", "user_id", "review_text"], keep="first"
                ).reset_index(drop=True)
                total_reviews += len(reviews)
            
            done_map = get_done_map(out_df)
            processed += 1
            last_saved_idx = actual_idx  # 마지막 처리한 인덱스 업데이트
            
            if not verbose:
                print(f"[인덱스 {actual_idx}] ok={ok} got={len(reviews)} product_rating={product_rating} url={url} err={err[:120]}")
            
            # ✅ 각 제품마다 즉시 저장 (중간에 멈춰도 데이터 손실 방지)
            try:
                out_df.to_csv(OUT_TABLE1, index=False, encoding="utf-8-sig")
                if verbose:
                    print(f"  💾 저장 완료 (인덱스 {actual_idx}까지, 총 {len(out_df)}개 리뷰)")
            except Exception as e:
                if verbose:
                    print(f"  ⚠️  저장 실패: {str(e)[:50]}")
            
            # 추가 체크포인트 메시지 (N개마다)
            if processed % SAVE_EVERY_PRODUCTS == 0:
                if verbose:
                    print(f"\n💾 [체크포인트] {processed}개 제품 처리 완료 (인덱스 {last_saved_idx}까지)")
                    print(f"   총 수집된 리뷰: {total_reviews}개")
            
            time.sleep(POLITE_SLEEP_SEC)
        
        # final save
        out_df.to_csv(OUT_TABLE1, index=False, encoding="utf-8-sig")
        browser.close()
    
    if verbose:
        print("\n" + "="*60)
        print("✅ 크롤링 완료!")
        print("="*60)
        print(f"📊 처리된 제품: {processed}개")
        print(f"📊 처리 범위: 인덱스 {start_idx}~{last_saved_idx} (마지막 처리: 인덱스 {last_saved_idx})")
        print(f"📝 총 수집된 리뷰: {total_reviews}개 (피부타입 필수, 유용한 순)")
        print(f"💾 저장 파일: {OUT_TABLE1}")
        print("="*60)
        print(f"\n💡 다음 크롤링 시 이어서 하려면:")
        print(f"   start_idx={last_saved_idx + 1}, end_idx={end_idx}")
        print("="*60)
    else:
        print(f"✅ Done: {OUT_TABLE1} rows={len(out_df)} (인덱스 {start_idx}~{last_saved_idx})")
    
    return out_df

if __name__ == "__main__":
    import sys
    
    # 명령줄 인자 처리
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    headless_mode = "--headless" in args
    verbose_mode = "--quiet" not in args  # --quiet가 없으면 verbose 모드
    test_mode = "--test" in args
    
    # start_idx, end_idx 파라미터 파싱
    start_idx = 0
    end_idx = None
    for arg in args:
        if arg.startswith("--start="):
            try:
                start_idx = int(arg.split("=", 1)[1])
            except ValueError:
                print(f"⚠️  잘못된 start_idx 값: {arg.split('=', 1)[1]}")
        elif arg.startswith("--end="):
            try:
                end_idx = int(arg.split("=", 1)[1])
            except ValueError:
                print(f"⚠️  잘못된 end_idx 값: {arg.split('=', 1)[1]}")
    
    # 테스트 모드
    if test_mode:
        print("\n🧪 테스트 모드 활성화")
        print("💡 사용법: python oliveyoung_reviews_crawler_helpful.py --test [--headless]")
        
        # URL 찾기
        test_url = None
        for arg in args:
            if arg.startswith("--url="):
                test_url = arg.split("=", 1)[1]
                break
        
        df_reviews = test_single_product(product_url=test_url, headless=headless_mode, verbose=verbose_mode, limit=120)
        
        if df_reviews is not None and len(df_reviews) > 0:
            # 테스트 결과를 임시 파일로 저장
            test_output = "test_reviews_output_helpful.csv"
            df_reviews.to_csv(test_output, index=False, encoding="utf-8-sig")
            print(f"\n💾 테스트 결과 저장: {test_output}")
    else:
        # 전체 크롤링 모드
        print("\n💡 팁: 브라우저를 보려면 headless=False로 설정하세요")
        print("   리뷰 수집 과정을 실시간으로 확인할 수 있습니다!")
        print("\n💡 테스트 모드: python oliveyoung_reviews_crawler_helpful.py --test")
        print("\n💡 범위 지정 사용법:")
        print("   python oliveyoung_reviews_crawler_helpful.py --start=0 --end=50")
        print("   python oliveyoung_reviews_crawler_helpful.py --start=50 --end=100")
        print("   (인덱스는 0부터 시작, end_idx는 포함되지 않음)\n")
        
        df_reviews = scrape_all_products_reviews(
            headless=headless_mode, 
            verbose=verbose_mode,
            start_idx=start_idx,
            end_idx=end_idx
        )
        
        if df_reviews is not None and len(df_reviews) > 0:
            print("\n" + "="*60)
            print("📋 결과 미리보기")
            print("="*60)
            print(df_reviews[["product_id", "product_name", "product_rating", "user_id", "user_rating", "user_keywords", "review_text"]].head(10))
            print(f"\n📊 총 리뷰 수: {len(df_reviews)}개")
            print("="*60)