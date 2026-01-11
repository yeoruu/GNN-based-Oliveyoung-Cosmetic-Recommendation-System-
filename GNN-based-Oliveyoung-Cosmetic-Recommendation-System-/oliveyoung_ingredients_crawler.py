import os
import re
import time
import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError

INPUT_TABLE2 = "table2_essence_basic.csv"   # 이미 만든 테이블2
OUT_TABLE3 = "table3_essence_ingredients.csv"

POLITE_SLEEP = 0.8          # 서버 예의(차단 방지)
RETRY = 2                   # 제품 단위 재시도 횟수
SAVE_EVERY = 20             # N개마다 체크포인트 저장

TARGET_ROW_LABEL = "화장품법에 따라 기재해야 하는 모든 성분"
TOGGLE_TITLE = "상품정보 제공고시"

def load_done_urls():
    """이미 크롤링한 URL 목록 로드"""
    if os.path.exists(OUT_TABLE3):
        df = pd.read_csv(OUT_TABLE3)
        done = set(df["product_url"].astype(str).tolist())
        return df, done
    else:
        df = pd.DataFrame(columns=["product_id", "product_url", "ingredients", "ok", "error"])
        return df, set()

def normalize_text(s: str) -> str:
    """텍스트 정규화 (공백 정리)"""
    if s is None:
        return None
    s = re.sub(r"\s+", " ", s).strip()
    return s

def open_and_extract(page, url: str) -> str:
    """
    1) 상세 페이지 접속
    2) '상품정보 제공고시' 섹션을 펼침(토글)
    3) 표에서 TARGET_ROW_LABEL 행을 찾아 td 텍스트 반환
    """
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_timeout(600)

    # ✅ 1) '상품정보 제공고시' 토글 펼치기
    # - 텍스트 기반으로 찾고 클릭 (사이트 구조 바뀌어도 살아남게)
    # - 이미 펼쳐져 있으면 클릭해도 큰 문제 없게 설계
    toggle = page.locator(f"text={TOGGLE_TITLE}").first
    if toggle.count() > 0:
        # 토글 클릭 가능하도록 보이게 이동
        try:
            toggle.scroll_into_view_if_needed(timeout=2000)
        except:
            page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.65);")
            page.wait_for_timeout(300)

        # 클릭 시도 (가끔 텍스트 자체가 아닌 상위 헤더가 클릭 대상일 수 있어 ancestor도 시도)
        clicked = False
        try:
            toggle.click(timeout=1500)
            clicked = True
        except:
            try:
                toggle.locator("xpath=ancestor::*[1]").click(timeout=1500)
                clicked = True
            except:
                pass

        if clicked:
            page.wait_for_timeout(500)

    # ✅ 2) 표에서 "화장품법에 따라 기재..." 행 찾기
    # 전략:
    #   - th에 TARGET_ROW_LABEL 포함하는 tr을 찾고
    #   - 그 tr의 td 텍스트를 읽는다.
    row = page.locator(f"xpath=//tr[.//th[contains(normalize-space(.), '{TARGET_ROW_LABEL}')]]").first
    if row.count() == 0:
        # fallback: 페이지 전체 텍스트에서 라벨이 보이는지 확인 후 좀 더 넓게 탐색
        # (테이블이 div 구조로 바뀌는 경우 대비)
        key = page.locator(f"text={TARGET_ROW_LABEL}").first
        if key.count() == 0:
            raise RuntimeError("TARGET_LABEL_NOT_FOUND")

        # key 주변에서 가장 가까운 td/내용 후보 탐색(최후)
        # - 같은 행의 다음 형제 요소 텍스트 등을 시도
        try:
            key.scroll_into_view_if_needed(timeout=1500)
        except:
            pass

        # 근처의 표 셀 후보(td)들 중 가장 긴 텍스트를 성분으로 가정
        tds = page.locator("td")
        best = ""
        for i in range(min(tds.count(), 80)):
            txt = normalize_text(tds.nth(i).inner_text())
            if txt and len(txt) > len(best) and ("," in txt or "정제수" in txt or "글리세" in txt):
                best = txt
        if not best:
            raise RuntimeError("INGREDIENTS_NOT_FOUND_FALLBACK")
        return best

    # 정상 케이스: 해당 tr의 td 가져오기
    cell = row.locator("td").first
    if cell.count() == 0:
        raise RuntimeError("TD_NOT_FOUND_IN_ROW")

    ingredients = normalize_text(cell.inner_text())
    if not ingredients:
        raise RuntimeError("EMPTY_INGREDIENTS")

    return ingredients

def scrape_ingredients(headless=True, verbose=True):
    """모든 제품의 성분 정보 크롤링"""
    if verbose:
        print("="*60)
        print("🧪 올리브영 성분 크롤러 시작")
        print("="*60)
        print(f"입력 파일: {INPUT_TABLE2}")
        print(f"출력 파일: {OUT_TABLE3}")
        print("="*60)
    
    # 테이블2 파일 로드
    if not os.path.exists(INPUT_TABLE2):
        print(f"❌ 오류: {INPUT_TABLE2} 파일을 찾을 수 없습니다.")
        return None
    
    t2 = pd.read_csv(INPUT_TABLE2)
    # 필요한 컬럼만
    t2 = t2[["product_id", "product_url"]].dropna().drop_duplicates(subset=["product_url"]).reset_index(drop=True)

    out_df, done_urls = load_done_urls()
    total = len(t2)
    
    if verbose:
        print(f"\n📊 총 제품 수: {total}개")
        print(f"📊 이미 완료된 제품: {len(done_urls)}개")
        print(f"📊 남은 제품: {total - len(done_urls)}개")
        print("="*60)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            locale="ko-KR",
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
            viewport={"width": 1280, "height": 900}
        )
        page = context.new_page()

        processed = 0
        success_count = 0
        fail_count = 0
        
        for idx, r in t2.iterrows():
            pid = str(r["product_id"])
            url = str(r["product_url"])

            if url in done_urls:
                if verbose:
                    print(f"[{idx+1}/{total}] ⏭️  건너뜀 (이미 완료): {url}")
                continue

            ok = False
            err = ""
            ingredients = None

            if verbose:
                print(f"\n[{idx+1}/{total}] 🔍 처리 중: {url}")

            for attempt in range(RETRY + 1):
                try:
                    ingredients = open_and_extract(page, url)
                    ok = True
                    err = ""
                    success_count += 1
                    if verbose:
                        print(f"  ✅ 성공! (시도 {attempt+1}/{RETRY+1})")
                        if ingredients:
                            print(f"  📋 성분 길이: {len(ingredients)}자")
                    break
                except Exception as e:
                    err = f"{type(e).__name__}:{str(e)}"
                    if verbose and attempt < RETRY:
                        print(f"  ⚠️  실패 (시도 {attempt+1}/{RETRY+1}): {err}")
                    # 가끔 팝업/로딩 꼬임 방지용 리로드
                    try:
                        page.wait_for_timeout(400)
                    except:
                        pass
                    if attempt < RETRY:
                        page.wait_for_timeout(800)
            
            if not ok:
                fail_count += 1
                if verbose:
                    print(f"  ❌ 최종 실패: {err}")

            out_df = pd.concat([out_df, pd.DataFrame([{
                "product_id": pid,
                "product_url": url,
                "ingredients": ingredients,
                "ok": ok,
                "error": err
            }])], ignore_index=True)

            done_urls.add(url)
            processed += 1

            # 체크포인트 저장
            if processed % SAVE_EVERY == 0:
                out_df.to_csv(OUT_TABLE3, index=False, encoding="utf-8-sig")
                if verbose:
                    print(f"\n💾 [체크포인트 저장] {processed}개 처리 완료 -> {OUT_TABLE3}")
                    print(f"   성공: {success_count}개, 실패: {fail_count}개")

            if not verbose:
                print(f"[{len(done_urls)}/{total}] ok={ok} url={url} err={err[:80]}")
            
            time.sleep(POLITE_SLEEP)

        # final save
        out_df.to_csv(OUT_TABLE3, index=False, encoding="utf-8-sig")
        browser.close()

    if verbose:
        print("\n" + "="*60)
        print("✅ 크롤링 완료!")
        print("="*60)
        print(f"📊 총 처리: {processed}개")
        print(f"✅ 성공: {success_count}개")
        print(f"❌ 실패: {fail_count}개")
        print(f"💾 저장 파일: {OUT_TABLE3}")
        print("="*60)
    else:
        print(f"✅ Done. Saved: {OUT_TABLE3}")
    
    return out_df

if __name__ == "__main__":
    import sys
    
    # 명령줄 인자로 headless 모드 제어 (기본값: False - 브라우저 보이기)
    headless_mode = "--headless" in sys.argv if len(sys.argv) > 1 else False
    verbose_mode = "--quiet" not in sys.argv  # --quiet가 없으면 verbose 모드
    
    print("\n💡 팁: 브라우저를 보려면 headless=False로 설정하세요")
    print("   성분 추출 과정을 실시간으로 확인할 수 있습니다!\n")
    
    df3 = scrape_ingredients(headless=headless_mode, verbose=verbose_mode)
    
    if df3 is not None and len(df3) > 0:
        print("\n" + "="*60)
        print("📋 결과 미리보기")
        print("="*60)
        # 성공한 항목만 미리보기
        success_df = df3[df3["ok"] == True]
        if len(success_df) > 0:
            print(success_df[["product_id", "ok", "ingredients"]].head(10))
            print(f"\n📊 성분 추출 성공: {len(success_df)}개")
        else:
            print("⚠️  성공한 항목이 없습니다.")
        
        # 실패한 항목 확인
        fail_df = df3[df3["ok"] == False]
        if len(fail_df) > 0:
            print(f"\n❌ 성분 추출 실패: {len(fail_df)}개")
            print("실패 원인:")
            print(fail_df["error"].value_counts())
        print("="*60)

