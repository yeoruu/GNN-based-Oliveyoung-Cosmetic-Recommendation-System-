import csv
import time
import random
from urllib.parse import urlparse, parse_qs
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import re

# --- [설정] ---
CONFIG = {
    "MAX_LIST_PAGES": None,         
    "MAX_PRODUCTS_PER_PAGE": None,  
    "MAX_REVIEWS_PER_PRODUCT": 100,
    "MIN_REVIEW_COUNT": 100
}

TARGET_URL = "https://www.oliveyoung.co.kr/store/display/getMCategoryList.do?dispCatNo=100000100010014&isLoginCnt=1&aShowCnt=0&bShowCnt=0&cShowCnt=0&trackingCd=Cat100000100010014_MID&trackingCd=Cat100000100010014_MID&t_page=%EC%B9%B4%ED%85%8C%EA%B3%A0%EB%A6%AC%EA%B4%80&t_click=%EC%B9%B4%ED%85%8C%EA%B3%A0%EB%A6%AC%EC%83%81%EC%84%B8_%EC%A4%91%EC%B9%B4%ED%85%8C%EA%B3%A0%EB%A6%AC&t_1st_category_type=%EB%8C%80_%EC%8A%A4%ED%82%A8%EC%BC%80%EC%96%B4&t_2nd_category_type=%EC%A4%91_%EC%97%90%EC%84%BC%EC%8A%A4%2F%EC%84%B8%EB%9F%BC%2F%EC%95%B0%ED%94%8C"

CATEGORY_NAME = "에센스/세럼/앰플"

FILE_REVIEWS = "table1_reviews.csv"
FILE_PRODUCTS = "table2_products.csv"

review_index = 1
product_index = 1

def extract_goods_no(url):
    try:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        return qs.get("goodsNo", [""])[0]
    except:
        return ""

def save_review(data):
    global review_index
    with open(FILE_REVIEWS, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["review_id", "product_id", "user_id", "user_keywords", "review_text", "user_rating"])
        data["review_id"] = f"E{review_index:03d}"
        review_index += 1
        writer.writerow(data)

def save_product(data):
    global product_index
    with open(FILE_PRODUCTS, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["product_id", "category", "brand", "product_name", "ingredient", "product_rating"])
        data["product_id"] = f"E{product_index:03d}"
        product_index += 1
        writer.writerow(data)

def close_popups(page):
    try:
        page.evaluate("""() => {
            const selectors = ['.layer_pop', '.popup_close', '#ntoday_close', '.btn_today_close'];
            selectors.forEach(sel => document.querySelectorAll(sel).forEach(el => el.remove()));
        }""")
    except:
        pass

def wait_for_antibot(page):
    try:
        max_wait = 60
        waited = 0
        while waited < max_wait:
            is_loading = page.locator("text='잠시만 기다려 주세요'").count() > 0 or \
                         page.locator("text='확인 중'").count() > 0 or \
                         page.locator("text='비정상적인 접근'").count() > 0
            if is_loading:
                if waited == 0:
                    print("\n    [!] 보안 화면 감지...")
                time.sleep(1)
                waited += 1
            else:
                if waited > 0:
                    print("    [OK] 보안 해제")
                    time.sleep(2)
                return
        print("\n!!! [경고] 보안 화면 해제 실패 !!!")
        input(">>> 수동 해결 후 [Enter]...")
    except:
        pass

def get_product_rating(page):
    """평점 추출 - span.rating 기반"""
    try:
        # 방법 1: span.rating 클래스로 직접 찾기
        rating = page.evaluate("""() => {
            const ratingSpan = document.querySelector('span.rating');
            if (ratingSpan) return ratingSpan.innerText.trim();
            return null;
        }""")
        if rating:
            match = re.search(r'(\d+\.\d+)', rating)
            if match:
                print(f"    -> 평점: {match.group(1)}")
                return match.group(1)
    except:
        pass
    
    try:
        # 방법 2: 전체 텍스트에서 패턴 찾기
        rating = page.evaluate("""() => {
            const text = document.body.innerText;
            const match = text.match(/[★☆]\s*(\d+\.\d+)/);
            return match ? match[1] : null;
        }""")
        if rating:
            print(f"    -> 평점 (패턴): {rating}")
            return rating
    except:
        pass
    
    print("    -> 평점 없음")
    return "N/A"

def get_review_count(page):
    """리뷰 개수 확인"""
    try:
        count = page.evaluate("""() => {
            const text = document.body.innerText;
            const match = text.match(/리뷰\s*([\d,]+)/);
            return match ? parseInt(match[1].replace(/,/g, '')) : 0;
        }""")
        return count
    except:
        return 0

def get_ingredients(page):
    """성분 추출 - 개선"""
    try:
        print("    -> 상품정보 제공고시 찾는 중...")
        
        # 먼저 해당 영역까지 스크롤
        page.mouse.wheel(0, 5000)
        time.sleep(2)
        
        # 상품정보 제공고시 버튼 클릭 (여러 방법 시도)
        clicked = page.evaluate("""() => {
            // 방법 1: 버튼 텍스트로 찾기
            let buttons = Array.from(document.querySelectorAll('button, div[role="button"], a'));
            let btn = buttons.find(b => {
                const text = b.innerText || b.textContent || '';
                return text.includes('상품정보 제공고시') || text.includes('상품정보') || text.includes('제공고시');
            });
            if (btn) { 
                btn.scrollIntoView({behavior: 'smooth', block: 'center'});
                btn.click(); 
                return true; 
            }
            
            // 방법 2: Accordion 버튼 찾기
            const accordions = document.querySelectorAll('[class*="ccordion"], [class*="tab"], [class*="detail"]');
            for (let acc of accordions) {
                const text = acc.innerText || acc.textContent || '';
                if (text.includes('상품정보')) {
                    acc.click();
                    return true;
                }
            }
            
            return false;
        }""")
        
        if not clicked:
            print("    -> 상품정보 제공고시 버튼 못 찾음, 전체 HTML 검색 시도...")
            # 버튼 없어도 이미 열려있을 수 있음
        else:
            print("    -> 버튼 클릭 완료")
        
        time.sleep(3)  # 충분한 대기
        
        # 성분 찾기 - 여러 방법 시도
        ingredients = page.evaluate("""() => {
            // 방법 1: 모든 테이블 순회
            const tables = document.querySelectorAll('table');
            for (let table of tables) {
                const rows = table.querySelectorAll('tr');
                for (let row of rows) {
                    const th = row.querySelector('th');
                    if (th) {
                        const thText = th.innerText || th.textContent || '';
                        // "화장품법에 따라" 또는 "모든 성분" 찾기
                        if (thText.includes('화장품법에 따라') || thText.includes('모든 성분') || thText.includes('전성분')) {
                            const td = row.querySelector('td');
                            if (td) {
                                const text = td.innerText || td.textContent || '';
                                return text.trim();
                            }
                        }
                    }
                }
            }
            
            // 방법 2: "성분" 키워드 근처 텍스트 찾기
            const allElements = document.querySelectorAll('*');
            for (let el of allElements) {
                if (el.children.length === 0) { // 텍스트만 있는 요소
                    const text = el.innerText || el.textContent || '';
                    if (text.length > 50 && (text.includes('워터') || text.includes('글리세린') || text.includes('알코올'))) {
                        // 성분처럼 보이는 텍스트
                        return text.trim();
                    }
                }
            }
            
            return null;
        }""")
        
        if ingredients and len(ingredients) > 20:
            print(f"    -> 성분 수집 성공: {len(ingredients)}자")
            return ingredients
        
        # 실패 시 전체 HTML에서 패턴으로 찾기 (최후의 수단)
        try:
            html = page.content()
            # "화장품법에 따라" 다음에 나오는 <td> 태그 찾기
            match = re.search(r'화장품법에 따라.*?<td[^>]*>(.*?)</td>', html, re.DOTALL)
            if match:
                from bs4 import BeautifulSoup
                text = BeautifulSoup(match.group(1), 'html.parser').get_text(strip=True)
                if len(text) > 20:
                    print(f"    -> 성분 수집 성공 (HTML): {len(text)}자")
                    return text
        except:
            pass
        
        print("    -> 성분 못 찾음")
        return "정보없음"
        
    except Exception as e:
        print(f"    -> 성분 에러: {e}")
        return "정보없음"

def get_detail_info(page, url, goods_no):
    ingredients = "정보없음"
    product_rating = "N/A"
    
    try:
        print(f"    -> 페이지 로딩 중...")
        page.goto(url, wait_until="domcontentloaded", timeout=180000)  # 180초(3분)로 증가
        wait_for_antibot(page)
        print("    -> 초기 로딩 대기 중...")
        time.sleep(8)  # 8초로 증가
        close_popups(page)
        
        # networkidle 대기 (에러 무시)
        try:
            print("    -> 네트워크 안정화 대기 중...")
            page.wait_for_load_state("networkidle", timeout=90000)  # 90초로 증가
            print("    -> 네트워크 안정화 완료")
        except:
            print("    -> networkidle 타임아웃, 강제 대기 중...")
            time.sleep(10)  # 타임아웃 시 10초 추가 대기
        
        time.sleep(5)  # 최종 안정화 5초
        print("    -> 로딩 완료!")

        # 1. 평점 수집
        product_rating = get_product_rating(page)
        
        # 2. 리뷰 개수 확인
        review_count = get_review_count(page)
        print(f"    -> 리뷰 개수: {review_count}개")
        
        if review_count < CONFIG["MIN_REVIEW_COUNT"]:
            print(f"    [스킵] 리뷰 {CONFIG['MIN_REVIEW_COUNT']}개 미만")
            return None, None

        # 3. 성분 수집
        ingredients = get_ingredients(page)
        
        # 4. 리뷰 탭으로 이동 - URL 직접 접근 방식
        try:
            print("    -> 리뷰 탭으로 이동 중 (URL 방식)...")
            
            # 방법 1: URL에 tab=review 파라미터 추가해서 직접 이동
            review_url = url if '?' in url else url + '?'
            if 'tab=' not in review_url:
                review_url += '&tab=review' if '?' in url else 'tab=review'
            
            print(f"    -> 리뷰 URL로 이동: {review_url}")
            page.goto(review_url, wait_until="domcontentloaded", timeout=120000)
            time.sleep(8)
            wait_for_antibot(page)
            
            print("    -> 리뷰 페이지 로딩 완료")
            time.sleep(5)
            
            # "등록된 리뷰가 없어요" 확인 및 재시도
            for retry in range(5):  # 3번 → 5번으로 증가
                no_review = page.locator("text='등록된 리뷰가 없어요'").count() > 0
                
                if no_review:
                    print(f"    [재시도 {retry + 1}/5] 리뷰 없음 메시지 감지...")
                    
                    if retry < 4:  # 마지막 시도 전까지만 새로고침
                        # 스크롤 먼저 (사람처럼)
                        for _ in range(3):
                            page.mouse.wheel(0, random.randint(2000, 4000))
                            time.sleep(random.uniform(0.5, 1.5))
                        
                        time.sleep(3)  # 추가 대기
                        
                        # 새로고침 대신 탭 다시 클릭 시도
                        print(f"    -> 리뷰 탭 다시 클릭 시도...")
                        page.evaluate("""() => {
                            const btns = Array.from(document.querySelectorAll('button'));
                            const btn = btns.find(b => b.innerText && b.innerText.includes('리뷰'));
                            if (btn) btn.click();
                        }""")
                        time.sleep(10)  # 충분한 로딩 시간
                    else:
                        print(f"    -> 마지막 시도, 최대 대기 중...")
                        time.sleep(15)  # 마지막엔 아주 오래 기다림
                else:
                    print("    -> 리뷰 로딩 확인!")
                    break
            
            # 리뷰 영역까지 스크롤 + 리뷰 감지
            print("    -> 리뷰 영역까지 스크롤 중...")
            review_found = False
            for scroll_num in range(10):  # 최대 10번
                page.mouse.wheel(0, 3000)
                time.sleep(1)
                
                # 스크롤마다 리뷰 있는지 확인
                review_count_now = page.locator("oy-review-review-item").count()
                if review_count_now > 0:
                    print(f"    -> 스크롤 {scroll_num + 1}번째에서 리뷰 {review_count_now}개 발견!")
                    review_found = True
                    break
            
            if not review_found:
                print("    -> 스크롤 완료, 추가 대기 중...")
                time.sleep(3)
            else:
                print("    -> 리뷰 발견, 수집 시작!")
                time.sleep(2)  # 짧은 안정화
            
            # 리뷰 요소 최종 확인
            try:
                if not review_found:
                    page.wait_for_selector("oy-review-review-item", timeout=30000)
                    print("    -> oy-review-review-item 감지!")
                else:
                    print("    -> 이미 리뷰 로딩됨, 바로 수집 시작")
            except:
                print("    [경고] oy-review-review-item 타임아웃, 계속 진행...")
                # 디버깅: 스크린샷 저장
                try:
                    screenshot_path = f"debug_review_{goods_no}.png"
                    page.screenshot(path=screenshot_path, full_page=True)
                    print(f"    -> 디버그 스크린샷 저장: {screenshot_path}")
                except:
                    pass

            # 실제 리뷰 요소 개수 확인
            review_count_check = page.locator("oy-review-review-item").count()
            print(f"    -> 실제 리뷰 요소 개수: {review_count_check}개")
            
            if review_count_check == 0:
                # 다른 selector 시도
                alt_selectors = [
                    "div.review-item",
                    "[class*='review-item']",
                    "[class*='ReviewItem']",
                    "li[class*='review']"
                ]
                for selector in alt_selectors:
                    alt_count = page.locator(selector).count()
                    if alt_count > 0:
                        print(f"    -> 대체 selector '{selector}' 발견: {alt_count}개")
                        break

            # 5. 리뷰 수집
            total_collected = 0
            collected_hashes = set()
            user_review_count = {}
            no_new_data_count = 0

            while total_collected < CONFIG["MAX_REVIEWS_PER_PRODUCT"]:
                review_hosts = page.locator("oy-review-review-item")
                count = review_hosts.count()
                
                if count == 0:
                    print("    [경고] 리뷰 요소 0개")
                    break
                
                newly_added = 0

                for i in range(count):
                    if total_collected >= CONFIG["MAX_REVIEWS_PER_PRODUCT"]:
                        break

                    try:
                        host = review_hosts.nth(i)
                        
                        data = host.evaluate("""(el) => {
                            const root = el.shadowRoot;
                            if (!root) return null;

                            let userId = '익명';
                            let keywords = '';
                            let rating = 'N/A';
                            let content = '';

                            // 사용자 정보
                            const userComp = root.querySelector('oy-review-review-user');
                            if (userComp && userComp.shadowRoot) {
                                const userRoot = userComp.shadowRoot;
                                // 사용자 이름 찾기
                                const nameEl = userRoot.querySelector('.name, [class*="name"]');
                                if (nameEl) userId = nameEl.innerText.trim();
                            }

                            // 키워드 - review-item 직접 하위에 있음!
                            const tagsDiv = root.querySelector('.tags, div.tags');
                            if (tagsDiv) {
                                // 모든 텍스트 노드와 요소 가져오기
                                const allText = tagsDiv.innerText || tagsDiv.textContent || '';
                                keywords = allText.trim().replace(/\\s+/g, '·');
                                
                                // 또는 span들 찾기
                                if (!keywords) {
                                    const spans = tagsDiv.querySelectorAll('span, [class*="tag"]');
                                    if (spans.length > 0) {
                                        keywords = Array.from(spans).map(s => s.innerText.trim()).filter(Boolean).join('·');
                                    }
                                }
                            }

                            // 별점 - oy-review-star-icon 개수
                            const stars = root.querySelectorAll('oy-review-star-icon');
                            if (stars.length > 0) rating = stars.length.toString();

                            // 리뷰 내용
                            const contentComp = root.querySelector('oy-review-review-content');
                            if (contentComp && contentComp.shadowRoot) {
                                const contentRoot = contentComp.shadowRoot;
                                const p = contentRoot.querySelector('p, div, .content');
                                if (p) content = p.innerText.trim();
                            }
                            
                            content = content.replace(/[\\n\\r]+/g, ' ');
                            if (!content) content = "[내용없음]";

                            return { userId, keywords, rating, content };
                        }""")

                        if data and data['keywords']:
                            user_id = data['userId']
                            
                            # "TOP XXX" 제거
                            import re
                            user_id = re.sub(r'\s*TOP\s+\d+\s*', '', user_id).strip()
                            
                            # 동일 사용자 2회 이상 스킵
                            if user_id in user_review_count:
                                user_review_count[user_id] += 1
                                if user_review_count[user_id] > 1:
                                    continue
                            else:
                                user_review_count[user_id] = 1
                            
                            key = (user_id, data['content'])
                            if key not in collected_hashes:
                                collected_hashes.add(key)
                                save_review({
                                    "product_id": goods_no,
                                    "user_id": user_id,
                                    "user_keywords": data['keywords'],
                                    "review_text": data['content'],
                                    "user_rating": data['rating']
                                })
                                total_collected += 1
                                newly_added += 1
                    except:
                        continue
                
                print(f"    -> {total_collected}개 수집 (이번 턴 +{newly_added})")

                if total_collected >= CONFIG["MAX_REVIEWS_PER_PRODUCT"]:
                    break
                
                # 스크롤
                page.mouse.wheel(0, 20000)  # 10000 → 20000 (더 많이 스크롤)
                time.sleep(0.5)
                page.keyboard.press("End")
                time.sleep(3)  # 2초 → 3초 (더 많은 로딩 시간)
                wait_for_antibot(page)

                if newly_added == 0:
                    no_new_data_count += 1
                    if no_new_data_count >= 10:  # 5번 → 10번 (더 참을성 있게)
                        print("    [종료] 더 이상 리뷰 없음")
                        break
                else:
                    no_new_data_count = 0

        except Exception as e:
            print(f"    [리뷰 로직 에러] {e}")

    except Exception as e:
        print(f"    [페이지 에러] {e}")
        return None, None
    
    return ingredients, product_rating

def main():
    with open(FILE_REVIEWS, "w", newline="", encoding="utf-8-sig") as f:
        csv.DictWriter(f, fieldnames=["review_id", "product_id", "user_id", "user_keywords", "review_text", "user_rating"]).writeheader()

    with open(FILE_PRODUCTS, "w", newline="", encoding="utf-8-sig") as f:
        csv.DictWriter(f, fieldnames=["product_id", "category", "brand", "product_name", "ingredient", "product_rating"]).writeheader()

    with sync_playwright() as p:
        print(">> 브라우저 실행...")
        browser = p.chromium.launch(
            headless=False, 
            channel="chrome", 
            args=[
                "--start-maximized",
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox"
            ]
        )
        context = browser.new_context(
            viewport=None,
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        list_page_idx = 1
        print(f"=== 수집 시작 (제품당 {CONFIG['MAX_REVIEWS_PER_PRODUCT']}개, 리뷰 {CONFIG['MIN_REVIEW_COUNT']}개 이상) ===")
        
        while True:
            target_url = f"{TARGET_URL}&pageIdx={list_page_idx}&rowsPerPage=24"
            try:
                page.goto(target_url, wait_until="domcontentloaded", timeout=120000)
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

                print(f"\n=== {list_page_idx}페이지: {len(product_items)}개 상품 ===")

                for item in product_items:
                    try:
                        info_box = item.select_one("div.prd_info")
                        if not info_box: continue

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
                        
                        if not goods_no: continue

                        detail_url = f"https://www.oliveyoung.co.kr/store/goods/getGoodsDetail.do?goodsNo={goods_no}"
                        print(f"\n[제품] {name}")

                        ingredients, product_rating = get_detail_info(page, detail_url, goods_no)
                        
                        if ingredients is None:
                            continue

                        save_product({
                            "product_id": goods_no,
                            "category": CATEGORY_NAME,
                            "brand": brand,
                            "product_name": name,
                            "ingredient": ingredients,
                            "product_rating": product_rating
                        })
                        
                        # 제품 간 대기 (봇 탐지 방지)
                        time.sleep(random.uniform(3, 6))

                    except Exception as e:
                        print(f"    [상품 에러] {e}")
                        continue

                if CONFIG["MAX_LIST_PAGES"] and list_page_idx >= CONFIG["MAX_LIST_PAGES"]:
                    break
                
                list_page_idx += 1

            except Exception as e:
                print(f"  [메인 에러] {e}")
                input(">>> 에러 해결 후 [Enter]...")

        browser.close()
        print(f"\n=== 완료 ===")
        print(f"리뷰: {FILE_REVIEWS}")
        print(f"제품: {FILE_PRODUCTS}")

if __name__ == "__main__":
    main()