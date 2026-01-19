import os
import re
import time
import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError

CATEGORY = "크림"
START_URL = "https://www.oliveyoung.co.kr/store/display/getMCategoryList.do?dispCatNo=100000100010014&isLoginCnt=0&aShowCnt=0&bShowCnt=0&cShowCnt=0&gateCd=Drawer&trackingCd=Cat100000100010014_MID&trackingCd=Cat100000100010014_MID&t_page=%EB%93%9C%EB%A1%9C%EC%9A%B0_%EC%B9%B4%ED%85%8C%EA%B3%A0%EB%A6%AC&t_click=%EC%B9%B4%ED%85%8C%EA%B3%A0%EB%A6%AC%ED%83%AD_%EC%A4%91%EC%B9%B4%ED%85%8C%EA%B3%A0%EB%A6%AC&t_1st_category_type=%EB%8C%80_%EC%8A%A4%ED%82%A8%EC%BC%80%EC%96%B4&t_2nd_category_type=%EC%A4%91_%EC%97%90%EC%84%BC%EC%8A%A4%2F%EC%84%B8%EB%9F%BC%2F%EC%95%B0%ED%94%8C"
OUT_CSV = "table2_cream_basic.csv"

MAX_SCROLL_PER_PAGE = 30      
SCROLL_WAIT_MS = 650
POLITE_SLEEP_SEC = 0.7        

def normalize_url(href: str) -> str:
    """URL 정규화"""
    if href.startswith("/"):
        return "https://www.oliveyoung.co.kr" + href
    return href

def load_existing():
    """기존 CSV 파일이 있으면 로드"""
    if os.path.exists(OUT_CSV):
        df = pd.read_csv(OUT_CSV)
        seen = set(df["product_url"].astype(str).tolist())
        return df, seen
    return pd.DataFrame(columns=["product_id", "category", "brand", "product_name", "product_url"]), set()

def click_view_48(page):
    """VIEW 48 있으면 클릭"""
    try:
        page.locator("text=48").first.click(timeout=1200)
        page.wait_for_timeout(800)
    except:
        pass

def scroll_for_loading_products(page, max_scroll=MAX_SCROLL_PER_PAGE, verbose=False):
    """
    ✅ 상품 lazy-load를 끝까지 붙이기 위한 스크롤
    """
    if verbose:
        print(f"  📜 상품 로딩을 위한 스크롤 시작 (최대 {max_scroll}회)...")
    
    last_cnt = -1
    stable = 0
    for i in range(max_scroll):
        page.mouse.wheel(0, 2800)
        page.wait_for_timeout(SCROLL_WAIT_MS)

        cnt = page.locator('a[href*="getGoodsDetail.do"]').count()
        if cnt == last_cnt:
            stable += 1
        else:
            stable = 0
            last_cnt = cnt

        if stable >= 2:
            if verbose:
                print(f"  ✅ 상품 로딩 완료 ({i+1}회 스크롤, {cnt}개 상품 발견)")
            break
    
    if verbose and stable < 2:
        print(f"  ✅ 스크롤 완료 ({max_scroll}회, {last_cnt}개 상품 발견)")

def scroll_to_pagination_bottom(page):
    """
    ✅ 페이지 이동을 위해 '맨 아래 페이지네이션'이 화면에 보이도록 끝까지 내려감
    """
    page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
    page.wait_for_timeout(700)

def extract_products_on_page(page, seen_urls: set, verbose=False):
    """현재 페이지에서 제품 정보 추출"""
    rows = []
    cards = page.locator('a[href*="getGoodsDetail.do"]')
    n = cards.count()
    
    if verbose:
        print(f"  🔍 발견된 상품 카드: {n}개")

    for i in range(n):
        a = cards.nth(i)
        href = a.get_attribute("href")
        if not href:
            continue

        product_url = normalize_url(href)
        if product_url in seen_urls:
            continue

        container = a.locator("xpath=ancestor::li[1]")

        brand = None
        for sel in [".tx_brand", ".prd_brand", ".brand"]:
            loc = container.locator(sel)
            if loc.count() > 0:
                t = loc.first.inner_text().strip()
                if t:
                    brand = t
                    break

        name = None
        for sel in [".tx_name", ".prd_name", ".name"]:
            loc = container.locator(sel)
            if loc.count() > 0:
                t = loc.first.inner_text().strip()
                if t and len(t) >= 2:
                    name = t
                    break

        if not name:
            try:
                name = a.inner_text().strip()
            except:
                name = None

        rows.append({
            "brand": brand,
            "product_name": name,
            "category": CATEGORY,
            "product_url": product_url
        })
        seen_urls.add(product_url)

    if verbose:
        new_count = len(rows)
        duplicate_count = n - new_count
        print(f"  📦 새로 수집된 제품: {new_count}개 (중복 제외: {duplicate_count}개)")

    return rows

def get_current_page_num(page) -> int:
    """페이지네이션에서 현재 페이지 찾기"""
    for sel in ["div.pageing strong", "strong.on", "a.on", "span.on", "a.active", "span.active", "strong"]:
        loc = page.locator(sel)
        if loc.count() > 0:
            txt = loc.first.inner_text().strip()
            if txt.isdigit():
                return int(txt)

    try:
        nums = page.locator("div.pageing a").filter(has_text=re.compile(r"^\d+$"))
        if nums.count() > 0:
            return int(nums.first.inner_text().strip())
    except:
        pass
    
    try:
        nums = page.locator("a, button").filter(has_text=re.compile(r"^\d+$"))
        if nums.count() > 0:
            for i in range(nums.count()):
                elem = nums.nth(i)
                classes = elem.get_attribute("class") or ""
                if "on" in classes or "active" in classes or "current" in classes:
                    txt = elem.inner_text().strip()
                    if txt.isdigit():
                        return int(txt)
            txt = nums.first.inner_text().strip()
            if txt.isdigit():
                return int(txt)
    except:
        pass
    
    raise RuntimeError("현재 페이지 번호를 찾지 못했습니다.")

def click_page_number(page, target: int, verbose=False) -> bool:
    """
    ✅ 맨 아래 페이지네이션에서 target 숫자를 클릭하여 페이지 이동
    ✅ 10, 20, 30... 단위 넘어갈 때 "다음" 버튼 자동 클릭
    """
    if verbose:
        print(f"    🔍 {target}번 페이지 링크 찾는 중...")
    
    scroll_to_pagination_bottom(page)

    # 먼저 현재 페이지가 target인지 확인 (strong 태그)
    try:
        strong = page.locator("strong[title='현재 페이지']")
        if strong.count() > 0:
            current_text = strong.first.inner_text().strip()
            if current_text == str(target):
                if verbose:
                    print(f"    ✅ 이미 {target}번 페이지에 있음 (strong 태그)")
                return True  # 이미 해당 페이지에 있으므로 성공
    except:
        pass

    loc = None
    try:
        # 방법 1: data-page-no 속성으로 찾기 (a 태그만 해당)
        temp_loc = page.locator(f"a[data-page-no='{target}']")
        if temp_loc.count() > 0:
            loc = temp_loc.first
            if verbose:
                print(f"    ✅ {target}번 페이지 발견 (data-page-no)")
    except:
        pass
    
    if loc is None:
        try:
            # 방법 2: div.pageing 안의 a 태그
            temp_loc = page.locator("div.pageing a").filter(has_text=re.compile(rf"^{target}$"))
            if temp_loc.count() > 0:
                txt = temp_loc.first.inner_text().strip()
                if txt == str(target):
                    loc = temp_loc.first
                    if verbose:
                        print(f"    ✅ {target}번 페이지 링크 발견 (div.pageing a)")
        except:
            pass
    
    if loc is None:
        try:
            # 방법 3: 전체 a 중 숫자 찾기
            temp_loc = page.locator("a").filter(has_text=re.compile(rf"^{target}$"))
            if temp_loc.count() > 0:
                txt = temp_loc.first.inner_text().strip()
                if txt == str(target):
                    loc = temp_loc.first
                    if verbose:
                        print(f"    ✅ {target}번 페이지 링크 발견 (전체 검색)")
        except:
            pass

    # ✅ 링크가 없으면 "다음" 버튼 클릭 시도 (10, 20, 30... 넘어갈 때)
    if loc is None:
        if verbose:
            print(f"    ⚠️  {target}번 페이지 링크가 보이지 않음, '다음' 버튼 시도...")
        
        next_clicked = False
        # 올리브영의 정확한 "다음" 버튼 클래스
        for next_selector in [
            "a.pageing_next",  # ✅ 올리브영 "다음" 버튼
            "a[class*='next']",
            "a[class*='Next']",
            "button.pageing_next",
            "a:has-text('다음')",
            "a:has-text('›')", 
            "a:has-text('>')",
            "button:has-text('다음')",
            "button:has-text('›')"
        ]:
            try:
                next_btn = page.locator(next_selector)
                if next_btn.count() > 0:
                    if verbose:
                        print(f"    👉 '다음' 버튼 발견! (selector: {next_selector})")
                    next_btn.first.click(timeout=2000)
                    page.wait_for_timeout(1500)
                    next_clicked = True
                    break
            except Exception as e:
                if verbose:
                    print(f"    ⚠️  {next_selector} 클릭 실패: {e}")
                continue
        
        if not next_clicked:
            if verbose:
                print(f"    ❌ '다음' 버튼도 찾을 수 없습니다.")
            return False
        
        # "다음" 버튼 클릭 후 충분히 대기
        if verbose:
            print(f"    ⏳ 페이지 번호 로딩 대기 중 (5초)...")
        page.wait_for_timeout(5000)  # 1500ms → 5000ms
        
        # 페이지네이션 다시 보이게
        scroll_to_pagination_bottom(page)
        page.wait_for_timeout(1000)
        
        try:
            # data-page-no 속성으로 다시 찾기
            temp_loc = page.locator(f"[data-page-no='{target}']")
            if temp_loc.count() > 0:
                loc = temp_loc.first
                if verbose:
                    print(f"    ✅ '다음' 버튼 클릭 후 {target}번 페이지 발견! (data-page-no)")
        except:
            pass
        
        if loc is None:
            try:
                temp_loc = page.locator("div.pageing a").filter(has_text=re.compile(rf"^{target}$"))
                if temp_loc.count() > 0:
                    txt = temp_loc.first.inner_text().strip()
                    if txt == str(target):
                        loc = temp_loc.first
                        if verbose:
                            print(f"    ✅ '다음' 버튼 클릭 후 {target}번 페이지 발견!")
            except:
                pass
        
        if loc is None:
            if verbose:
                print(f"    ❌ '다음' 버튼 클릭 후에도 {target}번 페이지를 찾을 수 없습니다.")
            return False

    before_href = None
    try:
        before_href = page.locator('a[href*="getGoodsDetail.do"]').first.get_attribute("href")
    except:
        pass

    try:
        if verbose:
            print(f"    👆 {target}번 페이지 클릭 중...")
        
        # JavaScript로 직접 클릭 (더 안정적)
        page.evaluate(f"""
            const elem = document.querySelector('[data-page-no="{target}"]');
            if (elem) {{
                elem.scrollIntoView({{behavior: 'smooth', block: 'center'}});
                elem.click();
            }}
        """)
        page.wait_for_timeout(1000)
        
    except Exception as e:
        if verbose:
            print(f"    ❌ {target}번 페이지 클릭 실패: {e}")
        return False

    if verbose:
        print(f"    ⏳ 페이지 로딩 대기 중...")
    for wait_count in range(30):
        page.wait_for_timeout(250)
        try:
            now_href = page.locator('a[href*="getGoodsDetail.do"]').first.get_attribute("href")
            if before_href and now_href and now_href != before_href:
                if verbose:
                    print(f"    ✅ 페이지 로딩 완료 ({wait_count * 0.25:.1f}초 소요)")
                return True
        except:
            pass

    if verbose:
        print(f"    ⚠️  페이지 변화 확인 불가, 계속 진행...")
    return True

def scrape_all(headless=True, verbose=True):
    """모든 페이지 크롤링"""
    df, seen = load_existing()
    
    if verbose:
        print("="*60)
        print("🚀 올리브영 크롤러 시작")
        print("="*60)
        print(f"카테고리: {CATEGORY}")
        print(f"시작 URL: {START_URL}")
        print(f"Headless 모드: {headless}")
        if len(df) > 0:
            print(f"기존 데이터: {len(df)}개 제품 발견")
        print("="*60)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
            locale="ko-KR"
        )

        if verbose:
            print("\n🌐 페이지 로딩 중...")
        page.goto(START_URL, wait_until="networkidle")
        
        if verbose:
            print("⚙️  VIEW 48 설정 시도...")
        click_view_48(page)
        if verbose:
            print("  ✅ VIEW 48 설정 완료 (없으면 건너뜀)")

        page_idx = 0

        while True:
            page_idx += 1
            if verbose:
                print(f"\n{'='*60}")
                print(f"📄 [페이지 {page_idx}] 크롤링 시작")
                print(f"{'='*60}")

            try:
                cur = get_current_page_num(page)
                if verbose:
                    print(f"📍 현재 페이지 번호: {cur}번")
            except Exception as e:
                if verbose:
                    print(f"⚠️  페이지 번호 확인 실패: {e}")
                cur = page_idx

            scroll_for_loading_products(page, verbose=verbose)

            if verbose:
                print("📋 제품 정보 추출 중...")
            rows = extract_products_on_page(page, seen, verbose=verbose)
            
            if rows:
                new_df = pd.DataFrame(rows)
                df = pd.concat([df, new_df], ignore_index=True)

            df = df.drop_duplicates(subset=["product_url"]).reset_index(drop=True)
            df["product_id"] = range(1, len(df) + 1)
            df = df[["product_id", "category", "brand", "product_name", "product_url"]]
            df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
            
            if verbose:
                print(f"\n💾 저장 완료!")
                print(f"  └─ 이번 페이지 신규 제품: {len(rows)}개")
                print(f"  └─ 총 누적 제품 수: {len(df)}개")
                print(f"  └─ 저장 파일: {OUT_CSV}")
            else:
                print(f"[Page {cur}] new={len(rows)} total={len(df)}")

            nxt = cur + 1
            if verbose:
                print(f"\n➡️  다음 페이지 이동 시도 ({cur}번 → {nxt}번)...")
            
            moved = click_page_number(page, nxt, verbose=verbose)
            if not moved:
                if verbose:
                    print("\n" + "="*60)
                    print("✅ 모든 페이지 크롤링 완료!")
                    print(f"  └─ 마지막 페이지: {cur}번")
                    print(f"  └─ 총 제품 수: {len(df)}개")
                    print("="*60)
                else:
                    print(f"[Done] 다음 페이지 번호를 못 찾아 종료합니다.")
                break

            time.sleep(POLITE_SLEEP_SEC)

        browser.close()

    if verbose:
        print(f"\n✅ 최종 완료: {OUT_CSV} / rows={len(df)}")
    return df

if __name__ == "__main__":
    import sys
    
    headless_mode = "--headless" in sys.argv if len(sys.argv) > 1 else False
    verbose_mode = "--quiet" not in sys.argv
    
    print("\n💡 팁: 브라우저를 보려면 headless=False로 설정하세요")
    print("   크롤링 진행 상황을 실시간으로 확인할 수 있습니다!\n")
    
    df2 = scrape_all(headless=headless_mode, verbose=verbose_mode)
    
    print("\n" + "="*60)
    print("📋 최종 결과 요약")
    print("="*60)
    print(df2.head(10))
    print(f"\n📊 총 상품 수: {len(df2)}개")
    print(f"💾 저장 파일: {OUT_CSV}")
    print("="*60)