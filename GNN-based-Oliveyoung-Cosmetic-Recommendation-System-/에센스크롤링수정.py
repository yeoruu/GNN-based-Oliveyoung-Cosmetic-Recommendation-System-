import csv
import time
import re
from urllib.parse import urlparse, parse_qs
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

# -----------------------------
# 설정
# -----------------------------
CONFIG = {
    "MAX_LIST_PAGES": None,            # None이면 끝까지
    "MAX_PRODUCTS_PER_PAGE": None,     # None이면 페이지 내 전부
    "TARGET_REVIEWS_PER_PRODUCT": 100, # "필터 통과 리뷰" 100개 확보
    "POLITE_SLEEP_SEC": 0.8,
}

TARGET_URL = "https://www.oliveyoung.co.kr/store/display/getMCategoryList.do?dispCatNo=100000100010014&fltDispCatNo=&prdSort=01&pageIdx=11&rowsPerPage=48&searchTypeSort=btn_thumb&plusButtonFlag=N&isLoginCnt=0&aShowCnt=0&bShowCnt=0&cShowCnt=0&trackingCd=Cat100000100010014_Small&amplitudePageGubun=&t_page=%EC%B9%B4%ED%85%8C%EA%B3%A0%EB%A6%AC%EA%B4%80&t_click=%EC%B9%B4%ED%85%8C%EA%B3%A0%EB%A6%AC%ED%83%AD_%EC%A4%91%EC%B9%B4%ED%85%8C%EA%B3%A0%EB%A6%AC&midCategory=%EC%97%90%EC%84%BC%EC%8A%A4%2F%EC%84%B8%EB%9F%BC%2F%EC%95%B0%ED%94%8F&smallCategory=%EC%A0%84%EC%B2%B4&checkBrnds=&lastChkBrnd=&t_1st_category_type=%EB%8C%80_%EC%8A%A4%ED%82%A8%EC%BC%80%EC%96%B4&t_2nd_category_type=%EC%A4%91_%EC%97%90%EC%84%BC%EC%8A%A4%2F%EC%84%B8%EB%9F%BC%2F%EC%95%B0%ED%94%8F"
# category는 dispCatNo가 고정이라면 고정값으로 두는 게 가장 안전(원하면 상세에서 breadcrumb로 파싱도 가능)
CATEGORY_NAME = "에센스/세럼/앰플"

FILE_TABLE1 = "table1_reviews.csv"
FILE_TABLE2 = "table2_products.csv"

SKIN_TYPES = ["지성", "건성", "복합성", "민감성", "약건성", "트러블성", "중성"]
SKIN_TYPES_SET = set(SKIN_TYPES)

# -----------------------------
# 유틸
# -----------------------------
def extract_goods_no(url):
    try:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        return qs.get("goodsNo", [""])[0]
    except:
        return ""

def clean_text(s):
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()

def close_popups(page):
    try:
        page.evaluate("""() => {
            const selectors = ['.layer_pop', '.popup_close', '#ntoday_close', '.btn_today_close'];
            selectors.forEach(sel => {
                const els = document.querySelectorAll(sel);
                els.forEach(el => el.remove());
            });
        }""")
    except:
        pass

def wait_for_antibot(page):
    try:
        max_wait = 30
        waited = 0
        while waited < max_wait:
            is_loading = (
                page.locator("text='잠시만 기다려 주세요'").count() > 0 or
                page.locator("text='확인 중'").count() > 0 or
                page.locator("text='비정상적인 접근'").count() > 0
            )
            if is_loading:
                if waited == 0:
                    print("\n    [!] 보안 대기 화면 감지... 대기 중...")
                time.sleep(1)
                waited += 1
            else:
                if waited > 0:
                    print("    [OK] 대기 화면 해제됨.")
                    time.sleep(1)
                return

        print("\n!!! [경고] 대기 화면이 사라지지 않습니다. !!!")
        input(">>> 해결 후 정상 화면이 나오면 여기를 클릭하고 [Enter]를 누르세요...")
    except:
        pass

def init_csv():
    with open(FILE_TABLE1, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["product_id", "user_id", "user_keywords", "review_text", "user_rating"]
        )
        writer.writeheader()

    with open(FILE_TABLE2, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["product_id", "category", "brand", "product_name", "ingredient", "product_rating"]
        )
        writer.writeheader()

def append_table1(row):
    with open(FILE_TABLE1, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["product_id", "user_id", "user_keywords", "review_text", "user_rating"]
        )
        writer.writerow(row)

def append_table2(row):
    with open(FILE_TABLE2, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["product_id", "category", "brand", "product_name", "ingredient", "product_rating"]
        )
        writer.writerow(row)

# -----------------------------
# 상세페이지: 성분 추출
# -----------------------------
def extract_ingredient(page):
    """
    '상품정보 제공고시' 열어서 전성분 텍스트 가져오기 (최대한 기존 코드 유지)
    """
    ingredient = "정보없음"
    try:
        page.evaluate("""() => {
            const btns = document.querySelectorAll('button, a');
            for (let b of btns) {
                if (b.innerText && b.innerText.includes('상품정보 제공고시')) {
                    b.click();
                    break;
                }
            }
        }""")
        page.wait_for_timeout(600)

        # 전성분
        ing_node = page.locator("xpath=//*[contains(text(), '전성분')]/following-sibling::*[1]")
        if ing_node.count() > 0:
            ingredient = clean_text(ing_node.first.inner_text())
        else:
            # 모든 성분
            ing_node = page.locator("xpath=//*[contains(text(), '모든 성분')]/following-sibling::*[1]")
            if ing_node.count() > 0:
                ingredient = clean_text(ing_node.first.inner_text())
    except:
        pass

    return ingredient or "정보없음"

# -----------------------------
# 리뷰 탭/정렬/평점 추출
# -----------------------------
def goto_reviews_tab(page):
    # 네가 쓰던 방식 + 조금 더 폭넓게
    clicked = page.evaluate("""() => {
        const buttons = Array.from(document.querySelectorAll("button, a"));
        const prefer = buttons.find(b => b.innerText && (b.innerText.includes('리뷰') && b.innerText.includes('(')));
        if (prefer) { prefer.click(); return true; }

        const tabs = Array.from(document.querySelectorAll("button[class*='GoodsDetailTabs_tab-item']"));
        const reviewBtn = tabs.find(b => b.innerText && b.innerText.includes('리뷰'));
        if (reviewBtn) { reviewBtn.click(); return true; }

        const fallback = buttons.find(b => b.innerText && b.innerText.includes('리뷰'));
        if (fallback) { fallback.click(); return true; }
        return false;
    }""")
    if not clicked:
        raise RuntimeError("REVIEWS_TAB_NOT_FOUND")

def set_sort_useful(page):
    # "유용한 순" 클릭
    candidates = [
        'button:has-text("유용한 순")',
        'a:has-text("유용한 순")',
        'text=유용한 순',
        '*:has-text("유용한 순")',
    ]
    for sel in candidates:
        loc = page.locator(sel)
        if loc.count() > 0:
            try:
                loc.first.scroll_into_view_if_needed(timeout=2000)
                loc.first.click(timeout=2000)
                page.wait_for_timeout(900)
                return
            except:
                continue
    # 못 찾아도 진행은 하되 로그
    print("    [!] '유용한 순' 버튼을 찾지 못했습니다. (계속 진행)")

def extract_product_rating(page):
    """
    리뷰 영역의 큰 평점(예: 4.7) 추출 (업로드된 파일의 방식 참고)
    """
    # 1) .rating-score
    try:
        loc = page.locator(".rating-score").first
        if loc.count() > 0:
            t = clean_text(loc.inner_text())
            m = re.search(r"([0-5]\.\d)", t)
            if m:
                return m.group(1)
    except:
        pass

    # 2) star-container 내부
    try:
        loc = page.locator(".star-container .rating-score").first
        if loc.count() > 0:
            t = clean_text(loc.inner_text())
            m = re.search(r"([0-5]\.\d)", t)
            if m:
                return m.group(1)
    except:
        pass

    # 3) HTML fallback
    try:
        html = page.content()
        m = re.search(r"rating-score[^>]*>([0-5]\.\d)", html)
        if m:
            return m.group(1)
    except:
        pass

    return ""

def click_more_if_exists(page) -> bool:
    before = page.locator("oy-review-review-item").count()
    for sel in ['button:has-text("더보기")', 'a:has-text("더보기")', 'text=더보기']:
        loc = page.locator(sel)
        if loc.count() > 0:
            try:
                loc.first.scroll_into_view_if_needed(timeout=2000)
                loc.first.click(timeout=2000)
                page.wait_for_timeout(400)
                page.mouse.wheel(0, 1200)
                page.wait_for_timeout(900)
                after = page.locator("oy-review-review-item").count()
                return after > before or True
            except:
                continue
    return False

def extract_user_rating_from_card(card):
    """
    리뷰 카드 내 사용자 별점 추출:
    - oy-review-star-icon 내부 svg path fill="#FF5753" 갯수
    (업로드된 파일 방식 차용)
    """
    rating = card.locator("div.rating").first
    if rating.count() == 0:
        rating = card

    stars = rating.locator("oy-review-star-icon")
    n = stars.count()
    if n == 0:
        return ""

    filled = 0
    for i in range(n):
        star = stars.nth(i)
        try:
            path = star.locator("svg path").first
            fill = path.get_attribute("fill")
            if fill and fill.lower() == "#ff5753":
                filled += 1
        except:
            pass

    if 0 <= filled <= 5:
        return str(filled)
    return ""

def parse_review_card(card):
    """
    user_id, user_keywords, user_rating, review_text
    - user_keywords: 원본에서 가져오되, 최종 저장은 피부타입 7개만 필터링해서 저장
    """
    # user_id
    user_id = ""
    try:
        name_loc = card.locator("div.name").first
        if name_loc.count() > 0:
            user_id = clean_text(name_loc.inner_text())
    except:
        pass

    # user_keywords 후보 추출
    user_keywords_raw = ""
    try:
        skin_types_div = card.locator("div.skin-types").first
        if skin_types_div.count() > 0:
            spans = skin_types_div.locator("span.skin-type")
            kws = []
            for i in range(spans.count()):
                t = clean_text(spans.nth(i).inner_text())
                if t:
                    kws.append(t)
            user_keywords_raw = " | ".join(kws)
    except:
        pass

    # 사용자 평점
    user_rating = extract_user_rating_from_card(card)

    # 리뷰 본문
    review_text = ""
    try:
        content_loc = card.locator("oy-review-review-content .content p").first
        if content_loc.count() > 0:
            review_text = clean_text(content_loc.inner_text())
        if not review_text:
            content_loc = card.locator(".review-content-container .content p").first
            if content_loc.count() > 0:
                review_text = clean_text(content_loc.inner_text())
        if not review_text:
            content_loc = card.locator("oy-review-review-content").first
            if content_loc.count() > 0:
                review_text = clean_text(content_loc.inner_text())
    except:
        pass

    return user_id, user_keywords_raw, user_rating, review_text

def filter_skin_keywords(user_keywords_raw: str) -> str:
    """
    user_keywords_raw에서 피부타입 7개만 남기고 반환
    """
    if not user_keywords_raw:
        return ""
    # 구분자 다양할 수 있으니 단어 단위로도 검사
    tokens = re.split(r"[|,/·\s]+", user_keywords_raw)
    tokens = [t.strip() for t in tokens if t.strip()]
    filtered = [t for t in tokens if t in SKIN_TYPES_SET]
    # 중복 제거 + 원래 순서 유지
    seen = set()
    out = []
    for t in filtered:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return " | ".join(out)

# -----------------------------
# 제품 1개 처리
# -----------------------------
def collect_useful_reviews_with_skin_only(page, product_id, detail_url):
    """
    조건:
    - 유용한 순
    - 피부타입 7개 중 하나라도 포함되는 리뷰만
    - 동일 user_id 최대 2개까지
    - 최종적으로 필터 통과 리뷰 100개 이상 확보 가능해야 저장
    """
    page.goto(detail_url, wait_until="domcontentloaded", timeout=60000)
    wait_for_antibot(page)
    page.wait_for_timeout(1200)
    close_popups(page)

    # 성분
    ingredient = extract_ingredient(page)

    # 리뷰탭
    goto_reviews_tab(page)
    page.wait_for_timeout(1500)
    wait_for_antibot(page)

    # 유용한 순
    set_sort_useful(page)
    page.wait_for_timeout(1000)

    # 제품 전체 평점
    product_rating = extract_product_rating(page)

    # 수집 루프
    collected = []
    seen = set()                 # (user_id, review_text[:80]) 등으로 중복 방지
    user_cnt = {}                # user_id별 카운트 (최대 2)
    guard = 0
    max_guard = 35
    max_iterations = 250
    iteration = 0

    target = CONFIG["TARGET_REVIEWS_PER_PRODUCT"]

    while len(collected) < target and guard < max_guard and iteration < max_iterations:
        iteration += 1

        cards = page.locator("oy-review-review-item")
        n = min(cards.count(), 250)

        before_len = len(collected)
        newly = 0

        for i in range(n):
            if len(collected) >= target:
                break

            card = cards.nth(i)
            try:
                user_id, user_keywords_raw, user_rating, review_text = parse_review_card(card)
            except:
                continue

            user_id = user_id or "익명"
            review_text = clean_text(review_text)

            if not review_text or len(review_text) < 5:
                continue

            # 피부타입 7개만 필터링
            skin_keywords = filter_skin_keywords(user_keywords_raw)
            if not skin_keywords:
                continue  # "사용자 키워드 있는 것만" + "피부타입 7개" 만족 못함

            # 동일 유저 2개 초과 제외
            c = user_cnt.get(user_id, 0)
            if c >= 2:
                continue

            key = f"{user_id}::{review_text[:120]}"
            if key in seen:
                continue
            seen.add(key)

            # 통과
            user_cnt[user_id] = c + 1
            collected.append({
                "product_id": product_id,
                "user_id": user_id,
                "user_keywords": skin_keywords,
                "review_text": review_text,
                "user_rating": user_rating or ""
            })
            newly += 1

        print(f"      -> 필터 통과 리뷰 {len(collected)}개 (이번 턴 +{newly})")

        if len(collected) >= target:
            break

        # 더보기/스크롤로 로딩 유도
        progressed = click_more_if_exists(page)
        page.mouse.wheel(0, 2500)
        page.wait_for_timeout(900)
        page.keyboard.press("End")
        page.wait_for_timeout(1200)
        wait_for_antibot(page)

        if len(collected) == before_len:
            guard += 1
            if guard % 5 == 0:
                print(f"      [진행 없음] guard={guard}/{max_guard}")
        else:
            guard = 0

    # 100개 못 채우면 "리뷰 100개 이상인 것만" 조건 때문에 실패 처리
    if len(collected) < target:
        return ingredient, product_rating, []  # 실패 -> 저장 안 하게

    return ingredient, product_rating, collected

# -----------------------------
# 메인: 카테고리 리스트 돌면서 상품 처리
# -----------------------------
def main():
    init_csv()

    with sync_playwright() as p:
        print(">> 크롬 브라우저 실행...")
        browser = p.chromium.launch(
            headless=False,
            channel="chrome",
            args=["--start-maximized", "--disable-blink-features=AutomationControlled"]
        )
        context = browser.new_context(viewport=None, locale="ko-KR")
        page = context.new_page()

        list_page_idx = 1
        print(f"=== 수집 시작 (제품당 필터 통과 리뷰 {CONFIG['TARGET_REVIEWS_PER_PRODUCT']}개 확보) ===")
        print("※ 보안 화면이 뜨면 해결 후 Enter를 누르세요 ※")

        while True:
            target_url = f"{TARGET_URL}&pageIdx={list_page_idx}&rowsPerPage=24"
            try:
                page.goto(target_url, wait_until="domcontentloaded", timeout=60000)
                wait_for_antibot(page)
                close_popups(page)
                page.mouse.wheel(0, 1000)
                time.sleep(1)

                soup = BeautifulSoup(page.content(), "html.parser")
                product_items = soup.select("ul.cate_prd_list > li")
                if not product_items:
                    print("  > 리스트 끝")
                    break

                if CONFIG["MAX_PRODUCTS_PER_PAGE"]:
                    product_items = product_items[:CONFIG["MAX_PRODUCTS_PER_PAGE"]]

                print(f"  > {list_page_idx}페이지: {len(product_items)}개 상품 처리")

                for item in product_items:
                    try:
                        info_box = item.select_one("div.prd_info")
                        if not info_box:
                            continue

                        brand = info_box.select_one(".tx_brand").get_text(strip=True) if info_box.select_one(".tx_brand") else ""
                        name = info_box.select_one(".tx_name").get_text(strip=True) if info_box.select_one(".tx_name") else ""

                        goods_no = ""
                        a_tag = item.select_one("a")
                        if a_tag:
                            href = a_tag.get("href", "")
                            if "goodsNo=" in href:
                                goods_no = extract_goods_no("http://dummy.com" + href)
                            elif item.get("data-ref-goodsno"):
                                goods_no = item.get("data-ref-goodsno")

                        if not goods_no:
                            continue

                        detail_url = f"https://www.oliveyoung.co.kr/store/goods/getGoodsDetail.do?goodsNo={goods_no}"
                        print(f"    -> [제품] {name}")

                        ingredient, product_rating, reviews = collect_useful_reviews_with_skin_only(
                            page, goods_no, detail_url
                        )

                        # ✅ 리뷰 100개 못 모으면 저장 제외
                        if not reviews:
                            print("      [스킵] 피부타입 포함 리뷰 100개 미만")
                            continue

                        # table2 저장 (조건 만족한 제품만)
                        append_table2({
                            "product_id": goods_no,
                            "category": CATEGORY_NAME,
                            "brand": brand,
                            "product_name": name,
                            "ingredient": ingredient,
                            "product_rating": product_rating
                        })

                        # table1 저장 (최종 100개만 저장: 필요하면 reviews 전체 저장으로 바꿀 수 있음)
                        for r in reviews[:CONFIG["TARGET_REVIEWS_PER_PRODUCT"]]:
                            append_table1(r)

                        time.sleep(CONFIG["POLITE_SLEEP_SEC"])

                    except Exception as e:
                        print(f"    [상품 루프 에러] {e}")
                        continue

                if CONFIG["MAX_LIST_PAGES"] and list_page_idx >= CONFIG["MAX_LIST_PAGES"]:
                    break

                list_page_idx += 1

            except Exception as e:
                print(f"  [메인 루프 에러] {e}")
                input(">>> 에러 발생. 해결 후 [Enter]...")

        browser.close()
        print(f"\n=== 완료 ===")
        print(f"- {FILE_TABLE1}")
        print(f"- {FILE_TABLE2}")

if __name__ == "__main__":
    main()
