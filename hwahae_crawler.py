"""
HWAWHAE (화해) - Selenium 크롤러 템플릿

목표:

1) 카테고리 전체보기 -> 카테고리(검정 pill) -> 스킨케어 펼치기

2) 하위 카테고리(스킨/토너, 로션/에멀젼, ...) 선택

3) 제품 목록에서 제품 상세 URL 수집

4) 상세 페이지에서 제품 정보 + "화면에 보이는" 리뷰 몇 개(예: 5~10개) 수집

5) CSV 저장 + 체크포인트(중간 저장)



주의:

- 화해는 DOM 구조/클래스가 바뀔 수 있어 selector를 1~2번은 조정해야 할 수 있습니다.

- 아래 코드는 "텍스트 기반 XPath"를 많이 사용해 최대한 튼튼하게 만들었고,

  필요한 경우 개발자도구에서 더 안정적인 data-testid / aria-label / 고정 class로 바꾸는 걸 권장합니다.

"""



import re
import os
import time
import random
import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional



import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
# -----------------------------
# Config
# -----------------------------

START_URL = "https://www.hwahae.co.kr/rankings"

SUBCATEGORIES = ["스킨/토너", "로션/에멀젼", "에센스/앰플/세럼", "크림"]  # 필요 시 추가/수정

# 크롤링할 하위 카테고리 선택 (비어있으면 모든 카테고리 크롤링)
# 예: SELECTED_SUBCATEGORIES = ["스킨/토너"]  # 스킨/토너만 크롤링
# 예: SELECTED_SUBCATEGORIES = []  # 모든 카테고리 크롤링
SELECTED_SUBCATEGORIES = ["에센스/앰플/세럼"]  # 스킨/토너 카테고리의 모든 제품 수집

# 테스트 모드: True로 설정하면 한 제품만 크롤링 (테스트용)
TEST_MODE = False  # 테스트 후 False로 변경 (3개 제품 테스트를 위해 False로 설정)
MAX_PRODUCTS_PER_SUBCAT = 10000   # 하위 카테고리당 몇 개 제품까지 상세 수집할지 (모든 제품 수집을 위해 큰 값으로 설정)
MAX_REVIEW_SNIPPETS = 2           # 상세페이지에서 리뷰 몇 개만 뽑을지(화면에 보이는 범위)
SCROLL_MAX = 200                   # 목록 무한스크롤 로딩 시 스크롤 횟수(모든 제품 수집을 위해 증가)
CHECKPOINT_EVERY = 5               # 상세 N개마다 CSV 저장 (대량 크롤링 시 더 자주 저장하여 데이터 손실 방지)

# 크롤링 모드 설정
# "list_only": 랭킹에서 제품 목록(제품명, 브랜드, URL)만 수집
# "details_only": 저장된 제품 목록 파일을 읽어서 상세 정보만 수집
# "both": 두 단계 모두 실행 (기본값)
CRAWL_MODE = "list_only"  # "list_only", "details_only", "both"
# 제품 목록 파일 (1단계에서 저장, 2단계에서 읽기)
PRODUCT_LIST_FILE = "hwahae_product_list.csv"  # 제품 목록만 저장하는 파일
OUTFILE = "hwahae_products_sample.csv"  # 최종 상세 정보가 포함된 파일
HEADLESS = False  # 디버깅을 위해 False로 설정 (실행 시 브라우저가 보임)

# -----------------------------
# Utilities
# -----------------------------

def human_sleep(a=0.6, b=1.4):
    """더 자연스러운 랜덤 딜레이"""
    time.sleep(random.uniform(a, b))

def random_mouse_movement(driver):
    """마우스를 랜덤하게 움직여 사람처럼 보이게"""
    try:
        actions = ActionChains(driver)
        # 랜덤한 위치로 마우스 이동
        x_offset = random.randint(-100, 100)
        y_offset = random.randint(-100, 100)
        actions.move_by_offset(x_offset, y_offset).perform()
        human_sleep(0.1, 0.3)
        # 원래 위치로 복귀
        actions.move_by_offset(-x_offset, -y_offset).perform()

    except:
        pass

def natural_scroll(driver, target_y=None):

    """자연스러운 스크롤 (부드럽게 여러 단계로)"""

    if target_y is None:
        target_y = driver.execute_script("return document.body.scrollHeight")
    current_y = driver.execute_script("return window.pageYOffset")
    distance = target_y - current_y
    steps = max(5, int(abs(distance) / 200))  # 여러 단계로 나눠서 스크롤

    for i in range(steps):

        scroll_y = current_y + (distance / steps) * (i + 1)

        driver.execute_script(f"window.scrollTo(0, {scroll_y});")

        human_sleep(0.1, 0.3)

    

    # 마지막에 정확한 위치로

    driver.execute_script(f"window.scrollTo(0, {target_y});")



def make_driver(headless: bool = True) -> webdriver.Chrome:

    options = webdriver.ChromeOptions()

    if headless:

        options.add_argument("--headless=new")

    options.add_argument("--window-size=1400,900")

    options.add_argument("--no-sandbox")

    options.add_argument("--disable-dev-shm-usage")

    options.add_argument("--lang=ko-KR")

    options.add_argument("--disable-blink-features=AutomationControlled")

    

    # 봇 감지 회피를 위한 설정

    options.add_experimental_option("excludeSwitches", ["enable-automation"])

    options.add_experimental_option('useAutomationExtension', False)

    options.add_argument("--disable-blink-features=AutomationControlled")

    

    # 실제 브라우저 User-Agent 사용

    user_agents = [

        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",

        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",

        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",

    ]

    options.add_argument(f"user-agent={random.choice(user_agents)}")



    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    driver.implicitly_wait(2)

    

    # WebDriver 속성 숨기기

    driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {

        'source': '''

            Object.defineProperty(navigator, 'webdriver', {

                get: () => undefined

            })

        '''

    })

    

    return driver



def wait_clickable(driver, xpath: str, timeout: int = 15):

    return WebDriverWait(driver, timeout).until(EC.element_to_be_clickable((By.XPATH, xpath)))



def wait_present(driver, xpath: str, timeout: int = 15):

    return WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.XPATH, xpath)))



def safe_click(driver, elem):

    """자연스러운 클릭 (마우스 움직임 포함)"""

    try:

        # 요소가 보이도록 스크롤

        driver.execute_script("arguments[0].scrollIntoView({block:'center', behavior:'smooth'});", elem)

        human_sleep(0.5, 1.0)  # 스크롤 후 대기 시간 증가

        

        # 랜덤 마우스 움직임

        random_mouse_movement(driver)

        human_sleep(0.3, 0.7)

        

        # ActionChains를 사용한 자연스러운 클릭

        try:

            actions = ActionChains(driver)

            actions.move_to_element(elem).pause(random.uniform(0.1, 0.3)).click().perform()

        except:

            # ActionChains 실패 시 일반 클릭

            elem.click()

    except Exception:

        # JavaScript 클릭 시도

        driver.execute_script("arguments[0].click();", elem)

    

    human_sleep(0.5, 1.2)  # 클릭 후 대기 시간 증가



def open_new_tab(driver, url: str):

    """새 탭을 열고 자연스럽게 전환"""

    driver.execute_script("window.open(arguments[0], '_blank');", url)

    human_sleep(0.5, 1.0)  # 탭 열기 후 잠시 대기

    driver.switch_to.window(driver.window_handles[-1])

    human_sleep(1.5, 2.5)  # 탭 전환 후 페이지 로딩 대기 시간 증가



def close_tab_back(driver):

    """탭을 닫고 원래 탭으로 복귀 (자연스럽게)"""

    driver.close()

    human_sleep(0.5, 1.0)  # 탭 닫기 후 잠시 대기

    driver.switch_to.window(driver.window_handles[0])

    human_sleep(1.0, 2.0)  # 원래 탭으로 복귀 후 대기 시간 증가



def scroll_to_bottom(driver, max_scroll: int = 10):

    """자연스러운 스크롤 (부드럽게 여러 단계로)"""

    last_height = driver.execute_script("return document.body.scrollHeight")

    current_position = 0

    

    for _ in range(max_scroll):

        # 랜덤한 스크롤 거리 (사람처럼)

        scroll_amount = random.randint(300, 800)

        current_position += scroll_amount

        target_position = min(current_position, last_height)

        

        # 자연스러운 스크롤

        natural_scroll(driver, target_position)

        human_sleep(1.0, 2.0)  # 스크롤 후 대기 시간 증가

        

        # 랜덤 마우스 움직임

        if random.random() > 0.5:

            random_mouse_movement(driver)

        

        new_height = driver.execute_script("return document.body.scrollHeight")

        if new_height == last_height and current_position >= last_height:

            break

        last_height = new_height

        

        # 가끔 위로 조금 스크롤 (사람처럼)

        if random.random() > 0.7:

            scroll_up = random.randint(100, 300)

            driver.execute_script(f"window.scrollBy(0, -{scroll_up});")

            human_sleep(0.5, 1.0)





# -----------------------------

# Step 1) Navigate category UI

# -----------------------------

def click_category_all_view(driver):

    """

    '카테고리 전체보기' 버튼 클릭

    """

    # 여러 selector 옵션 시도

    xpath_options = [

        "//*[contains(text(),'카테고리 전체보기')]",

        "//*[contains(text(),'카테고리') and contains(text(),'전체')]",

        "//button[contains(text(),'카테고리')]",

        "//a[contains(text(),'카테고리')]",

        "//*[@class and contains(@class,'category')]",

        "//*[contains(@aria-label,'카테고리')]",

    ]

    

    # 페이지가 완전히 로드될 때까지 대기

    human_sleep(2.0, 3.0)

    

    # 스크롤을 내려서 버튼이 보이도록 시도

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    human_sleep(1.0, 1.5)

    driver.execute_script("window.scrollTo(0, 0);")
    human_sleep(0.5, 1.0)

    

    for xpath in xpath_options:

        try:

            btn = wait_clickable(driver, xpath, timeout=5)

            if btn and btn.is_displayed():

                safe_click(driver, btn)

                human_sleep()

                print(f"  [SUCCESS] Found category button with: {xpath[:50]}...")

                return

        except Exception as e:

            continue

    

    # 모든 selector 실패 시 스크린샷 저장 및 에러 메시지

    try:

        driver.save_screenshot("debug_category_button_not_found.png")

        print("  [WARNING] Category button not found. Screenshot saved as debug_category_button_not_found.png")

    except:

        pass

    

    raise Exception("Could not find '카테고리 전체보기' button. Please check the website structure or update selectors.")



def is_category_tab_selected(driver) -> bool:

    """카테고리별 탭이 선택되어 있는지 확인 (URL 기준으로만 확인)"""

    try:

        # URL로 확인 (가장 정확) - english_name=category가 반드시 있어야 함

        current_url = driver.current_url

        if "english_name=category" in current_url:

            print(f"  [DEBUG] Category tab is selected (URL contains 'english_name=category'): {current_url}")

            return True

        else:

            print(f"  [DEBUG] Category tab is NOT selected (URL: {current_url})")

            return False

    except:

        return False


def click_category_tab(driver):

    """

    '카테고리별' 탭 클릭 (급상승 탭에서 카테고리별 탭으로 이동)

    HTML 구조: <a href="rankings?english_name=category&theme_id=2"><span>카테고리별</span></a>

    """

    # 현재 URL 확인

    current_url_before = driver.current_url

    print(f"  [DEBUG] Current URL before clicking tab: {current_url_before}")

    

    # 이미 카테고리별 탭이 선택되어 있는지 확인 (URL에 english_name=category가 있어야 함)

    if is_category_tab_selected(driver):

        print("  [INFO] '카테고리별' tab is already selected (URL contains 'english_name=category')")

        return

    else:

        print("  [INFO] '카테고리별' tab is NOT selected, will click now...")

    

    xpath_options = [

        # 실제 HTML 구조에 맞는 selector (우선순위 높음)

        # a 태그를 직접 찾기 (href 속성 포함)

        "//a[@href and contains(@href,'english_name=category') and contains(@href,'theme_id=2')]",

        "//a[contains(@href,'english_name=category')]",

        # a 태그 내부의 span으로 찾기

        "//a[.//span[normalize-space()='카테고리별']]",

        "//a[contains(@class,'hds-flex') and contains(@class,'hds-flex-col') and .//span[normalize-space()='카테고리별']]",

        # span 태그로 찾기 (fallback)

        "//span[contains(@class,'hds-relative') and contains(@class,'hds-grow') and normalize-space()='카테고리별']",

        "//span[contains(@class,'leading-[42px]') and normalize-space()='카테고리별']",

        "//span[normalize-space()='카테고리별']",

        # 일반적인 fallback

        "//*[contains(text(),'카테고리별')]",

        "//a[contains(text(),'카테고리별')]",

    ]

    

    human_sleep(1.0, 1.5)

    

    for idx, xpath in enumerate(xpath_options, 1):

        try:

            print(f"  [DEBUG] Trying xpath option {idx}: {xpath[:80]}...")

            tab = wait_clickable(driver, xpath, timeout=5)

            if tab and tab.is_displayed():

                print(f"  [DEBUG] Found tab element, clicking...")

                safe_click(driver, tab)

                human_sleep(3.0, 4.0)  # 페이지 전환 후 충분한 대기 시간

                

                # URL이 변경되었는지 확인

                current_url_after = driver.current_url

                print(f"  [DEBUG] URL before: {current_url_before}")

                print(f"  [DEBUG] URL after: {current_url_after}")

                

                # 카테고리별 페이지로 이동했는지 확인

                if "english_name=category" in current_url_after or "theme_id=2" in current_url_after:

                    print(f"  [SUCCESS] Clicked '카테고리별' tab and navigated to category page")

                    print(f"  [SUCCESS] Current URL: {current_url_after}")

                    return

                elif current_url_before != current_url_after:

                    print(f"  [INFO] URL changed but doesn't contain category params, waiting more...")

                    human_sleep(2.0, 3.0)

                    # 다시 확인

                    final_url = driver.current_url

                    if "english_name=category" in final_url or "theme_id=2" in final_url:

                        print(f"  [SUCCESS] '카테고리별' tab is now selected (URL: {final_url})")

                        return

                    else:

                        print(f"  [WARNING] URL changed but still not category page: {final_url}")

                else:

                    print(f"  [WARNING] URL didn't change after clicking tab")

        except Exception as e:

            print(f"  [DEBUG] Exception with xpath {idx}: {e}")

            continue

    

    print("  [ERROR] '카테고리별' tab not found or failed to click!")

    print(f"  [DEBUG] Current page source snippet (searching for '카테고리별'):")

    try:

        page_text = driver.page_source

        if "카테고리별" in page_text:

            print("    - '카테고리별' text found in page source")

        else:

            print("    - '카테고리별' text NOT found in page source")

    except:

        pass


def click_black_category_all(driver):

    """

    팝업/패널 상단의 검정 pill '카테고리 전체' 클릭 (토글 열기)

    HTML 구조: <span class="hds-bg-gray-850 hds-text-white ...">카테고리 전체</span>

    """

    # 실제 HTML 구조에 맞는 selector (우선순위 높음)

    xpath_options = [

        # 검정색 배경과 흰색 텍스트를 가진 span

        "//span[contains(@class,'hds-bg-gray-850') and contains(@class,'hds-text-white') and normalize-space()='카테고리 전체']",

        "//span[contains(@class,'hds-bg-gray-850') and contains(@class,'hds-text-white') and contains(text(),'카테고리 전체')]",

        "//span[contains(@class,'hds-bg-gray-850') and contains(text(),'카테고리 전체')]",

        # 일반적인 selector (fallback)

        "(//*[normalize-space()='카테고리 전체'])[1]",

        "//*[contains(text(),'카테고리 전체')]",

        "//button[contains(text(),'전체')]",

        "//*[@class and (contains(@class,'all') or contains(@class,'전체'))]",

    ]

    

    human_sleep(1.0, 1.5)  # 팝업/패널이 열릴 시간 확보

    

    for xpath in xpath_options:
        try:
            btn = wait_clickable(driver, xpath, timeout=5)
            if btn and btn.is_displayed():
                safe_click(driver, btn)
                print(f"  [SUCCESS] Found '카테고리 전체' with: {xpath[:70]}...")

        
                # 토글이 열릴 때까지 충분히 대기
                human_sleep(2.0, 3.0)
                print("  [INFO] Waiting for category toggle to open...")

                

                # 스킨케어가 나타날 때까지 대기 (토글이 열렸는지 확인)
                try:
                    WebDriverWait(driver, 10).until(
                        lambda d: len(d.find_elements(By.XPATH, "//summary[.//span[contains(text(),'스킨케어')]]")) > 0 or
                                 len(d.find_elements(By.XPATH, "//*[contains(text(),'스킨케어')]")) > 0
                    )
                    print("  [SUCCESS] Category toggle opened, '스킨케어' is visible")
                except:
                    print("  [WARNING] '스킨케어' not found immediately, but continuing...")

                return

        except Exception as e:
            continue

    print("  [WARNING] '카테고리 전체' button not found, continuing anyway...")



def expand_skincare_in_panel(driver):
    """
    카테고리 목록에서 '스킨케어' 섹션 펼치기 (클릭)
    HTML 구조: <summary><span class="flex items-center gap-8">스킨케어</span></summary>
    """
    xpath_options = [
        # 실제 HTML 구조에 맞는 selector (우선순위 높음)
        # summary 태그를 직접 클릭
        "//summary[.//span[contains(@class,'flex') and contains(@class,'items-center') and contains(@class,'gap-8') and normalize-space()='스킨케어']]",
        "//summary[.//span[normalize-space()='스킨케어']]",
        "//summary[contains(.//span/text(),'스킨케어')]",
        # summary 내부의 span 클릭
        "//summary//span[contains(@class,'flex') and contains(@class,'items-center') and contains(@class,'gap-8') and normalize-space()='스킨케어']",
        "//summary//span[normalize-space()='스킨케어']",
        # 일반적인 selector (fallback)
        "//span[@class='flex items-center gap-8' and contains(text(),'스킨케어')]",
        "//span[contains(@class,'flex') and contains(@class,'items-center') and normalize-space()='스킨케어']",
        "//*[normalize-space()='스킨케어']",
        "//*[contains(text(),'스킨케어')]",
    ]

    
    human_sleep(1.0, 1.5)

    for xpath in xpath_options:
        try:
            elem = wait_clickable(driver, xpath, timeout=5)
            if elem and elem.is_displayed():
                safe_click(driver, elem)
                human_sleep(1.5, 2.0)  # 하위 카테고리가 나타날 시간
                print(f"  [SUCCESS] Found '스킨케어' with: {xpath[:70]}...")


                # 하위 카테고리가 나타났는지 확인

                try:
                    WebDriverWait(driver, 5).until(
                        lambda d: len(d.find_elements(By.XPATH, "//span[normalize-space()='에센스/앰플/세럼']")) > 0 or 
                                 len(d.find_elements(By.XPATH, "//label[.//span[normalize-space()='에센스/앰플/세럼']]")) > 0
                    )
                    print("  [SUCCESS] Subcategories are now visible")

                except:

                    print("  [WARNING] Subcategories not found immediately, but continuing...")

                

                return

        except Exception as e:

            continue

    

    print("  [WARNING] '스킨케어' section not found, continuing anyway...")



def select_subcategory_chip(driver, subcat_text: str):

    """

    하위 카테고리 칩(예: 스킨/토너) 클릭 후 랭킹 페이지로 이동 확인

    HTML 구조: <label><input type="checkbox"><span>스킨/토너</span></label>

    """

    # 현재 URL 저장 (변경 확인용)

    current_url_before = driver.current_url

    

    # 실제 HTML 구조에 맞는 selector (우선순위 높음)
    # HTML 구조: 
    # - <span class="hds-inline-flex ... hds-bg-white ...">스킨/토너</span> (직접 span)
    # - <label><input><span>로션/에멀젼</span></label> (label 안의 span)

    xpath_options = [

        # span 태그 직접 클릭 (스킨/토너 같은 경우)

        f"//span[contains(@class,'hds-inline-flex') and contains(@class,'hds-cursor-pointer') and contains(@class,'hds-rounded-8') and normalize-space()='{subcat_text}']",

        f"//span[contains(@class,'hds-cursor-pointer') and contains(@class,'hds-rounded-8') and normalize-space()='{subcat_text}']",

        f"//span[contains(@class,'hds-bg-white') and normalize-space()='{subcat_text}']",

        f"//span[normalize-space()='{subcat_text}']",

        # label 태그를 찾아서 클릭 (로션/에멀젼 같은 경우)

        f"//label[.//span[normalize-space()='{subcat_text}']]",

        f"//label[contains(.//span/text(),'{subcat_text}')]",

        # label 내부의 span 클릭

        f"//label//span[contains(@class,'hds-cursor-pointer') and normalize-space()='{subcat_text}']",

        f"//label//span[normalize-space()='{subcat_text}']",

        # checkbox를 통한 클릭

        f"//label[.//span[normalize-space()='{subcat_text}']]//input[@type='checkbox']",

        # 일반적인 fallback

        f"//*[self::button or self::a or self::div][normalize-space()='{subcat_text}']",

        f"//*[contains(text(),'{subcat_text}')]",

        f"//button[contains(text(),'{subcat_text}')]",

        f"//a[contains(text(),'{subcat_text}')]",

    ]

    

    human_sleep(1.0, 1.5)

    

    for xpath in xpath_options:

        try:

            chip = wait_clickable(driver, xpath, timeout=5)

            if chip and chip.is_displayed():

                safe_click(driver, chip)

                print(f"  [INFO] Clicked subcategory '{subcat_text}' using: {xpath[:60]}...")

                

                # 페이지 로딩 대기 (URL 변경 또는 페이지 새로고침 대기)

                human_sleep(2.0, 3.0)

                

                # URL이 변경되었는지 확인

                try:

                    WebDriverWait(driver, 10).until(

                        lambda d: d.current_url != current_url_before or 

                                 "rankings" in d.current_url.lower() or

                                 "category" in d.current_url.lower()

                    )

                except:

                    pass  # URL이 변경되지 않아도 계속 진행

                

                current_url_after = driver.current_url

                print(f"  [INFO] Current URL after click: {current_url_after}")

                

                # 랭킹 페이지인지 확인

                if "rankings" in current_url_after.lower():

                    print(f"  [SUCCESS] Navigated to rankings page for '{subcat_text}'")

                else:

                    print(f"  [WARNING] URL doesn't contain 'rankings', but continuing...")

                

                # 제품 목록이 로드될 때까지 추가 대기

                human_sleep(2.0, 3.0)

                

                return

        except Exception as e:

            continue

    

    print(f"  [WARNING] Could not find subcategory '{subcat_text}', trying to continue...")





# -----------------------------

# Step 2) Collect product links from list page

# -----------------------------

def collect_product_links(driver, limit: int) -> List[Dict[str, str]]:

    """

    현재 목록 화면에서 제품 상세 링크, 제품명, 브랜드 수집 (랭킹 순서대로)

    HTML 구조: 

    - 링크: <li class="bg-white..."><a href="goods/...">

    - 브랜드: <span class="hds-text-body-medium hds-text-gray-tertiary">에스네이처</span>

    - 제품명: <span class="hds-text-body-medium hds-text-gray-primary">아쿠아 오아시스 토너</span>

    """

    print(f"  [INFO] Collecting product links from: {driver.current_url}")

    

    # 페이지 상단으로 이동 (랭킹 1위부터 시작)

    driver.execute_script("window.scrollTo(0, 0);")

    human_sleep(1.0, 1.5)

    

    # 제품 목록이 로드될 때까지 대기

    try:

        # 제품 링크가 나타날 때까지 대기 (goods/ 패턴)

        WebDriverWait(driver, 10).until(

            lambda d: len(d.find_elements(By.XPATH, "//a[contains(@href,'goods/')]")) > 0

        )

    except:

        print("  [WARNING] Product links not found immediately, continuing anyway...")

    

    # 랭킹 순서대로 수집하기 위해 li 태그 순서대로 수집

    products = []

    last_height = driver.execute_script("return document.body.scrollHeight")

    scroll_count = 0

    no_new_products_count = 0  # 새로운 제품이 로드되지 않은 연속 횟수

    MAX_NO_NEW_PRODUCTS = 5  # 연속으로 새로운 제품이 없으면 중단 (더 많이 시도)

    collected_urls = set()  # 중복 체크를 위한 set (더 빠른 검색)

    

    while len(products) < limit and scroll_count < SCROLL_MAX:

        # 현재 DOM에서 제품 개수 확인 (스크롤 전)

        product_count_before = len(driver.find_elements(By.XPATH, "//li[contains(@class,'bg-white')]//a[contains(@href,'goods/')]"))

        products_before_scroll = len(products)

        

        # li 태그 내의 a 태그를 순서대로 수집 (랭킹 순서 유지)

        product_items = driver.find_elements(By.XPATH, "//li[contains(@class,'bg-white')]//a[contains(@href,'goods/')]")

        

        print(f"  [DEBUG] Scroll {scroll_count + 1}: Found {len(product_items)} product items in DOM, collected {len(products)} so far")

        

        for a in product_items:

            try:

                href = a.get_attribute("href")

                if not href:

                    continue

                

                # goods/ 패턴 확인 및 전체 URL로 변환

                if "goods/" in href:

                    # 상대 경로인 경우 전체 URL로 변환

                    if href.startswith("/"):

                        href = "https://www.hwahae.co.kr" + href

                    elif not href.startswith("http"):

                        href = "https://www.hwahae.co.kr/" + href.lstrip("/")

                    
                    # URL 정규화 (쿼리 파라미터 제거하여 중복 방지)
                    href_normalized = href.split("?")[0] if "?" in href else href
                    
                    # 중복 제거 (set 사용으로 빠른 검색)

                    if href_normalized in collected_urls:

                        continue

                    collected_urls.add(href_normalized)

                    

                    # 제품명과 브랜드 추출

                    product_name = None

                    brand = None

                    rank = None  # 랭킹 번호

                    

                    try:

                        # a 태그의 부모 li에서 브랜드와 제품명 찾기

                        li_parent = a.find_element(By.XPATH, "./ancestor::li[contains(@class,'bg-white')][1]")

                        

                        # 랭킹 번호 추출
                        # HTML 구조: <div class="hds-flex ... hds-w-[30px] ..."><div class="">86</div>...
                        try:
                            rank_elem = li_parent.find_element(By.XPATH, ".//div[contains(@class,'hds-w-[30px]')]//div[not(@class) or @class='']")
                            rank_text = rank_elem.text.strip()
                            if rank_text.isdigit():
                                rank = int(rank_text)
                        except:
                            # 다른 방법으로 랭킹 찾기 시도
                            try:
                                # 랭킹이 있는 div 찾기
                                rank_divs = li_parent.find_elements(By.XPATH, ".//div[contains(@class,'hds-flex') and contains(@class,'hds-items-center')]//div")
                                for div in rank_divs:
                                    text = div.text.strip()
                                    if text.isdigit() and len(text) <= 3:  # 랭킹은 보통 3자리 이하
                                        rank = int(text)
                                        break
                            except:
                                pass

                        

                        # 브랜드: <span class="hds-text-body-medium hds-text-gray-tertiary">

                        try:

                            brand_elem = li_parent.find_element(By.XPATH, ".//span[contains(@class,'hds-text-body-medium') and contains(@class,'hds-text-gray-tertiary')]")

                            brand = brand_elem.text.strip()

                        except:

                            pass

                        

                        # 제품명: <span class="hds-text-body-medium hds-text-gray-primary">

                        try:

                            name_elem = li_parent.find_element(By.XPATH, ".//span[contains(@class,'hds-text-body-medium') and contains(@class,'hds-text-gray-primary')]")

                            product_name = name_elem.text.strip()

                        except:

                            pass

                    except Exception as e:

                        print(f"  [DEBUG] Error extracting product name/brand: {e}")

                    

                    products.append({

                        "url": href_normalized,  # 정규화된 URL 사용

                        "product_name": product_name,

                        "brand": brand,

                        "rank": rank  # 랭킹 번호 추가

                    })

                    

                    if len(products) >= limit:

                        break

            except Exception as e:

                continue

        

        if len(products) >= limit:

            break

        

        # 새로운 제품이 추가되었는지 확인

        new_products_count = len(products) - products_before_scroll

        if new_products_count == 0:

            no_new_products_count += 1

            print(f"  [DEBUG] No new products found (count: {no_new_products_count}/{MAX_NO_NEW_PRODUCTS})")

        else:

            no_new_products_count = 0  # 새로운 제품이 있으면 카운터 리셋

            print(f"  [DEBUG] Found {new_products_count} new products, total: {len(products)}")

        

        # 연속으로 새로운 제품이 없으면 중단

        if no_new_products_count >= MAX_NO_NEW_PRODUCTS:

            print(f"  [INFO] No new products for {MAX_NO_NEW_PRODUCTS} consecutive scrolls, stopping...")

            # 마지막으로 한 번 더 확인 (제품이 로드 중일 수 있음)

            human_sleep(3.0, 4.0)

            final_items = driver.find_elements(By.XPATH, "//li[contains(@class,'bg-white')]//a[contains(@href,'goods/')]")

            print(f"  [DEBUG] Final check: {len(final_items)} items in DOM")

            break

        

        # 페이지 하단까지 스크롤

        last_height_before = driver.execute_script("return document.body.scrollHeight")

        natural_scroll(driver, last_height_before)

        human_sleep(2.0, 3.0)  # 스크롤 후 기본 대기

        

        # 제품이 로드될 때까지 명시적으로 대기 (제품 개수가 증가하거나 높이가 변할 때까지)

        max_wait_attempts = 10

        for wait_attempt in range(max_wait_attempts):

            current_height = driver.execute_script("return document.body.scrollHeight")

            current_product_count = len(driver.find_elements(By.XPATH, "//li[contains(@class,'bg-white')]//a[contains(@href,'goods/')]"))

            
            # 제품 개수가 증가했거나 높이가 변했으면 로딩 완료

            if current_product_count > product_count_before or current_height > last_height_before:

                print(f"  [DEBUG] Products loaded (attempt {wait_attempt + 1}): {current_product_count} items, height changed: {current_height != last_height_before}")

                human_sleep(1.0, 1.5)  # 추가 안정화 시간

                break

            else:

                human_sleep(0.5, 1.0)  # 아직 로딩 중이면 조금 더 대기

        else:

            print(f"  [WARNING] Products may not have loaded after {max_wait_attempts} attempts")

        

        # 가끔 마우스 움직임 추가

        if random.random() > 0.6:

            random_mouse_movement(driver)

        

        # 새로운 콘텐츠가 로드되었는지 확인

        new_height = driver.execute_script("return document.body.scrollHeight")

        last_height = new_height

        scroll_count += 1

    
    # 마지막으로 전체 페이지를 다시 스캔하여 빠진 제품이 없는지 확인
    print(f"\n  [INFO] Final scan: Checking for any missed products...")
    
    # 페이지 상단부터 하단까지 다시 스크롤하면서 모든 제품 수집
    driver.execute_script("window.scrollTo(0, 0);")
    human_sleep(1.0, 1.5)
    
    # 페이지 하단까지 여러 번 스크롤하여 모든 제품 로드
    for final_scroll in range(3):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        human_sleep(2.0, 3.0)
        
        # 제품이 로드될 때까지 대기
        for wait_attempt in range(5):
            current_count = len(driver.find_elements(By.XPATH, "//li[contains(@class,'bg-white')]//a[contains(@href,'goods/')]"))
            human_sleep(0.5, 1.0)
            new_count = len(driver.find_elements(By.XPATH, "//li[contains(@class,'bg-white')]//a[contains(@href,'goods/')]"))
            if new_count == current_count:
                break
    
    # 전체 제품 다시 수집
    final_product_items = driver.find_elements(By.XPATH, "//li[contains(@class,'bg-white')]//a[contains(@href,'goods/')]")
    print(f"  [DEBUG] Final scan found {len(final_product_items)} product items in DOM")
    
    # 랭킹 번호 기반으로 누락된 제품 찾기
    collected_ranks = {p.get("rank") for p in products if p.get("rank") is not None}
    all_ranks_in_dom = set()
    
    # 빠진 제품이 있는지 확인
    for a in final_product_items:
        try:
            href = a.get_attribute("href")
            if not href:
                continue
            
            if "goods/" in href:
                if href.startswith("/"):
                    href = "https://www.hwahae.co.kr" + href
                elif not href.startswith("http"):
                    href = "https://www.hwahae.co.kr/" + href.lstrip("/")
                
                href_normalized = href.split("?")[0] if "?" in href else href
                
                # 랭킹 번호 추출
                rank = None
                product_name = None
                brand = None
                
                try:
                    li_parent = a.find_element(By.XPATH, "./ancestor::li[contains(@class,'bg-white')][1]")
                    
                    # 랭킹 번호 추출
                    try:
                        rank_elem = li_parent.find_element(By.XPATH, ".//div[contains(@class,'hds-w-[30px]')]//div[not(@class) or @class='']")
                        rank_text = rank_elem.text.strip()
                        if rank_text.isdigit():
                            rank = int(rank_text)
                            all_ranks_in_dom.add(rank)
                    except:
                        try:
                            rank_divs = li_parent.find_elements(By.XPATH, ".//div[contains(@class,'hds-flex') and contains(@class,'hds-items-center')]//div")
                            for div in rank_divs:
                                text = div.text.strip()
                                if text.isdigit() and len(text) <= 3:
                                    rank = int(text)
                                    all_ranks_in_dom.add(rank)
                                    break
                        except:
                            pass
                    
                    # 브랜드와 제품명 추출
                    try:
                        brand_elem = li_parent.find_element(By.XPATH, ".//span[contains(@class,'hds-text-body-medium') and contains(@class,'hds-text-gray-tertiary')]")
                        brand = brand_elem.text.strip()
                    except:
                        pass
                    try:
                        name_elem = li_parent.find_element(By.XPATH, ".//span[contains(@class,'hds-text-body-medium') and contains(@class,'hds-text-gray-primary')]")
                        product_name = name_elem.text.strip()
                    except:
                        pass
                except:
                    pass
                
                # 누락된 제품인지 확인 (URL 또는 랭킹 번호 기준)
                is_missed = False
                if href_normalized not in collected_urls:
                    is_missed = True
                elif rank is not None and rank not in collected_ranks:
                    is_missed = True
                    print(f"  [INFO] Found product with missing rank: rank {rank} - {product_name} ({brand})")
                
                if is_missed:
                    # 빠진 제품 발견
                    products.append({
                        "url": href_normalized,
                        "product_name": product_name,
                        "brand": brand,
                        "rank": rank
                    })
                    collected_urls.add(href_normalized)
                    if rank is not None:
                        collected_ranks.add(rank)
                    print(f"  [INFO] Found missed product: rank {rank}, {product_name} ({brand})")
        except:
            continue
    
    # 랭킹 순서 체크 및 누락된 랭킹 찾기
    if collected_ranks:
        min_rank = min(collected_ranks)
        max_rank = max(collected_ranks)
        expected_ranks = set(range(min_rank, max_rank + 1))
        missing_ranks = expected_ranks - collected_ranks
        
        if missing_ranks:
            print(f"  [WARNING] Missing ranks detected: {sorted(missing_ranks)}")
            print(f"  [INFO] Expected ranks: {min_rank} to {max_rank}, found: {len(collected_ranks)} ranks")
            
            # 누락된 랭킹의 제품을 찾기 위해 해당 위치로 스크롤
            for missing_rank in sorted(missing_ranks):
                print(f"  [INFO] Searching for missing rank {missing_rank}...")
                # 해당 랭킹 위치로 스크롤 (대략적인 위치 계산)
                scroll_position = (missing_rank - min_rank) / (max_rank - min_rank) * driver.execute_script("return document.body.scrollHeight")
                driver.execute_script(f"window.scrollTo(0, {scroll_position});")
                human_sleep(2.0, 3.0)
                
                # 해당 위치의 제품 찾기
                nearby_items = driver.find_elements(By.XPATH, "//li[contains(@class,'bg-white')]//a[contains(@href,'goods/')]")
                for a in nearby_items:
                    try:
                        li_parent = a.find_element(By.XPATH, "./ancestor::li[contains(@class,'bg-white')][1]")
                        try:
                            rank_elem = li_parent.find_element(By.XPATH, ".//div[contains(@class,'hds-w-[30px]')]//div[not(@class) or @class='']")
                            rank_text = rank_elem.text.strip()
                            if rank_text.isdigit() and int(rank_text) == missing_rank:
                                # 누락된 랭킹의 제품 발견
                                href = a.get_attribute("href")
                                if href:
                                    if href.startswith("/"):
                                        href = "https://www.hwahae.co.kr" + href
                                    elif not href.startswith("http"):
                                        href = "https://www.hwahae.co.kr/" + href.lstrip("/")
                                    href_normalized = href.split("?")[0] if "?" in href else href
                                    
                                    if href_normalized not in collected_urls:
                                        # 제품 정보 추출
                                        product_name = None
                                        brand = None
                                        try:
                                            brand_elem = li_parent.find_element(By.XPATH, ".//span[contains(@class,'hds-text-body-medium') and contains(@class,'hds-text-gray-tertiary')]")
                                            brand = brand_elem.text.strip()
                                        except:
                                            pass
                                        try:
                                            name_elem = li_parent.find_element(By.XPATH, ".//span[contains(@class,'hds-text-body-medium') and contains(@class,'hds-text-gray-primary')]")
                                            product_name = name_elem.text.strip()
                                        except:
                                            pass
                                        
                                        products.append({
                                            "url": href_normalized,
                                            "product_name": product_name,
                                            "brand": brand,
                                            "rank": missing_rank
                                        })
                                        collected_urls.add(href_normalized)
                                        collected_ranks.add(missing_rank)
                                        print(f"  [SUCCESS] Found missing rank {missing_rank}: {product_name} ({brand})")
                                        break
                        except:
                            continue
                    except:
                        continue
    
    # 랭킹 순서로 정렬
    products_with_rank = [p for p in products if p.get("rank") is not None]
    products_without_rank = [p for p in products if p.get("rank") is None]
    products_with_rank.sort(key=lambda x: x.get("rank", 9999))
    products = products_with_rank + products_without_rank
    
    print(f"  [INFO] Collected {len(products)} products (in ranking order) after final scan")
    if collected_ranks:
        print(f"  [INFO] Rank range: {min(collected_ranks)} to {max(collected_ranks)}, total ranks: {len(collected_ranks)}")

    return products[:limit]





# -----------------------------

# Step 3) Parse product detail page

# -----------------------------

def extract_text_safe(driver, xpath: str) -> Optional[str]:

    try:

        elem = driver.find_element(By.XPATH, xpath)

        txt = elem.text.strip()

        return txt if txt else None

    except Exception:

        return None



def extract_rating_and_review_count(driver) -> Dict[str, Optional[float]]:

    """

    상세페이지에서 평점과 리뷰 수 추출

    HTML 구조: <span class="hds-text-display-xlarge text-gray-primary">4.73</span>

    """

    rating = None

    review_count = None

    

    try:

        # 평점 추출

        rating_elem = driver.find_element(By.XPATH, "//span[contains(@class,'hds-text-display-xlarge') and contains(@class,'text-gray-primary')]")

        rating_text = rating_elem.text.strip()

        if rating_text:

            rating = float(rating_text)

    except:

        pass

    

    # 리뷰 수는 리뷰 섹션에서 추출하거나 목록에서 가져올 수 있음

    try:

        # 목록 페이지에서 본 리뷰 수 패턴 시도

        body_text = driver.find_element(By.TAG_NAME, "body").text

        m = re.search(r"(\d\.\d{1,2})\s*\(?([\d,]+)\)?", body_text)

        if m and not review_count:

            review_count = int(m.group(2).replace(",", ""))

    except:

        pass

    

    return {"rating": rating, "review_count": review_count}



def extract_product_detail(driver) -> Dict[str, Optional[str]]:

    """

    제품 상세 정보 추출

    HTML 구조:

    - 제품명: <span class="hds-text-title-medium text-gray-primary">

    - 정가: <span class="hds-text-body-medium text-gray-primary">300ml+300ml / 53,000원</span>

    - 순위: <span class="hds-text-body-medium text-gray-primary text-left">스킨/토너 ・ 수분 1위</span>

    """

    # 제품명

    name = extract_text_safe(driver, "//span[contains(@class,'hds-text-title-medium') and contains(@class,'text-gray-primary')]")

    if not name:

        name = extract_text_safe(driver, "//h1") or extract_text_safe(driver, "(//h1|//h2)[1]")



    # 정가 정보 추출 (용량과 가격 포함)

    price_info = None

    price = None

    volume = None

    try:

        # 정가 섹션 찾기

        price_elem = driver.find_element(By.XPATH, "//span[contains(@class,'hds-text-body-medium') and contains(@class,'text-gray-primary') and contains(.,'원')]")

        price_info = price_elem.text.strip()

        

        # 가격 추출

        pm = re.search(r"([\d,]+)\s*원", price_info)

        if pm:

            price = pm.group(1).replace(",", "")

        

        # 용량 추출

        vm = re.search(r"(\d+(?:\+\d+)?)\s*(ml|mL|g|G)", price_info)

        if vm:

            volume = f"{vm.group(1)}{vm.group(2)}"

    except:

        # fallback: 전체 페이지에서 검색

        try:

            body_text = driver.find_element(By.TAG_NAME, "body").text

            pm = re.search(r"([\d,]+)\s*원", body_text)

            if pm:

                price = pm.group(1).replace(",", "")

            vm = re.search(r"(\d+(?:\.\d+)?)\s*(ml|mL|g|G)", body_text)

            if vm:

                volume = f"{vm.group(1)}{vm.group(2)}"

        except:

            pass



    # 순위 정보

    rank = None

    try:

        rank_elem = driver.find_element(By.XPATH, "//span[contains(@class,'hds-text-body-medium') and contains(@class,'text-gray-primary') and contains(@class,'text-left') and contains(.,'위')]")

        rank = rank_elem.text.strip()

    except:

        pass



    # 브랜드 추출
    # HTML 구조: <span class="hds-text-body-medium text-gray-tertiary ml-8"><a href="/search?q=에스네이처">에스네이처</a></span>

    brand = None

    try:

        # 실제 HTML 구조에 맞는 selector (우선순위 높음)

        brand = extract_text_safe(driver, "//span[contains(@class,'hds-text-body-medium') and contains(@class,'text-gray-tertiary') and contains(@class,'ml-8')]//a")

        if brand:

            print(f"  [DEBUG] Found brand from span: {brand}")

    except:

        pass

    

    if not brand:

        try:

            # /search?q= 링크에서 브랜드 추출

            brand = extract_text_safe(driver, "//a[contains(@href,'/search?q=')]")

            if brand:

                print(f"  [DEBUG] Found brand from search link: {brand}")

        except:

            pass

    

    if not brand:

        try:

            # text-gray-tertiary 클래스를 가진 span 내의 a 태그

            brand = extract_text_safe(driver, "//span[contains(@class,'text-gray-tertiary')]//a[contains(@href,'/search')]")

            if brand:

                print(f"  [DEBUG] Found brand from tertiary span: {brand}")

        except:

            pass

    

    if not brand:

        try:

            # 제품명에서 브랜드 추출 시도 (fallback)

            if name:

                parts = name.split()

                if parts:

                    brand = parts[0]

                    print(f"  [DEBUG] Extracted brand from product name: {brand}")

        except:

            pass

    

    if not brand:

        try:

            # 기존 fallback 방법

            brand = extract_text_safe(driver, "//*[contains(@href,'/brands')][1]") or extract_text_safe(driver, "//*[contains(text(),'브랜드')]/following::*[1]")

        except:

            pass



    rr = extract_rating_and_review_count(driver)



    return {

        "product_name": name,

        "brand": brand,

        "price_krw": price,

        "volume": volume,

        "rank": rank,

        "rating": rr["rating"],

        "review_count": rr["review_count"],

        "url": driver.current_url,

    }



def click_review_tab_if_exists(driver):

    """

    상세페이지에서 '리뷰' 탭이 있으면 클릭해 노출을 늘림

    """

    candidates = driver.find_elements(By.XPATH, "//*[contains(text(),'리뷰')]")

    for c in candidates:

        try:

            if c.is_displayed():

                safe_click(driver, c)

                human_sleep(0.8, 1.4)

                return

        except Exception:

            continue



def extract_review_snippets(driver, max_reviews: int = 2) -> List[Dict[str, Optional[str]]]:

    """

    리뷰 정보 수집 (제품당 2개)

    한 리뷰어의 리뷰에 좋아요와 아쉬워요가 함께 있음

    HTML 구조:

    - 리뷰 컨테이너: <li class="py-24">

    - 닉네임: <span class="hds-text-subtitle-medium text-gray-primary">BRWI</span>

    - 사용자 타입: <span class="hds-text-smalltext-large ml-2 text-gray-secondary">30대/건성/민감성</span>

    - 좋은점: <div class="flex items-start gap-x-8 mt-24"><img src="/svgs/good.svg"><span class="hds-text-body-medium text-gray-primary line-clamp-3">...</span></div>

    - 아쉬운점: <div class="flex items-start gap-x-8 mt-24"><img src="/svgs/bad.svg"><span class="hds-text-body-medium text-gray-primary line-clamp-3">...</span></div>

    """

    click_review_tab_if_exists(driver)



    # 리뷰 영역으로 스크롤 (더 많은 리뷰가 보이도록)

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.3);")

    human_sleep(1.5, 2.0)

    

    # 리뷰가 로드될 때까지 대기

    try:

        WebDriverWait(driver, 10).until(

            lambda d: len(d.find_elements(By.XPATH, "//li[contains(@class,'py-24')]")) > 0

        )

    except:

        print("  [WARNING] No review items found immediately, trying to scroll more...")

        # 리뷰가 없으면 더 스크롤

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.5);")

        human_sleep(2.0, 3.0)

    

    # 리뷰 항목이 2개 미만이면 더 스크롤

    review_items_count = len(driver.find_elements(By.XPATH, "//li[contains(@class,'py-24')]"))

    if review_items_count < max_reviews:

        print(f"  [DEBUG] Only found {review_items_count} review items, scrolling more to find {max_reviews}...")

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.6);")

        human_sleep(2.0, 3.0)

        review_items_count = len(driver.find_elements(By.XPATH, "//li[contains(@class,'py-24')]"))

        print(f"  [DEBUG] After additional scroll, found {review_items_count} review items")



    reviews = []

    

    # <li class="py-24"> 기준으로 리뷰 찾기 (더 정확한 XPath)

    review_items = driver.find_elements(By.XPATH, "//li[contains(@class,'py-24')]")

    
    print(f"  [DEBUG] Found {len(review_items)} review items, extracting up to {max_reviews} reviews...")

    
    # 최대 max_reviews 개수만큼 리뷰 수집 (충분히 확인하기 위해 더 많이 순회)

    processed_nicknames = set()  # 중복 제거를 위한 set

    for idx, review_item in enumerate(review_items[:max_reviews * 3]):  # 더 많이 확인

        if len(reviews) >= max_reviews:

            break

        try:

            # 닉네임 찾기

            nickname = None

            try:

                nickname_elem = review_item.find_element(By.XPATH, ".//span[contains(@class,'hds-text-subtitle-medium') and contains(@class,'text-gray-primary')]")

                nickname = nickname_elem.text.strip()

                if not nickname or nickname in processed_nicknames:

                    print(f"  [DEBUG] Skipping review {idx+1}: No nickname or duplicate")

                    continue

                processed_nicknames.add(nickname)

            except Exception as e:

                print(f"  [DEBUG] Could not find nickname in review {idx+1}: {e}")

                continue  # 닉네임이 없으면 이 리뷰는 건너뛰기

            

            # 사용자 타입 찾기

            user_type = None

            try:

                type_elem = review_item.find_element(By.XPATH, ".//span[contains(@class,'hds-text-smalltext-large') and contains(@class,'ml-2') and contains(@class,'text-gray-secondary')]")

                user_type = type_elem.text.strip()

            except:

                pass

            

            # 좋은점 찾기 (good.svg 이미지가 있는 div 내의 span)

            good_point = None

            try:

                # 방법 1: good.svg 이미지가 있는 div를 찾고 그 안의 span 찾기 (가장 안정적)

                good_div = review_item.find_element(By.XPATH, ".//div[contains(@class,'flex') and contains(@class,'items-start') and contains(@class,'gap-x-8')]//img[@src='/svgs/good.svg']/ancestor::div[contains(@class,'flex') and contains(@class,'items-start')][1]")

                good_spans = good_div.find_elements(By.XPATH, ".//span[contains(@class,'hds-text-body-medium') and contains(@class,'text-gray-primary') and contains(@class,'line-clamp-3')]")

                if len(good_spans) >= 1:

                    good_point = good_spans[0].text.strip()

            except:

                # 방법 2: 더 간단한 XPath로 시도

                try:

                    good_div = review_item.find_element(By.XPATH, ".//img[@src='/svgs/good.svg']/ancestor::div[contains(@class,'flex')][1]")

                    good_spans = good_div.find_elements(By.XPATH, ".//span[contains(@class,'line-clamp-3')]")

                    if len(good_spans) >= 1:

                        good_point = good_spans[0].text.strip()

                except:

                    # 방법 3: line-clamp-3 span 중 첫 번째 (fallback)

                    try:

                        good_spans = review_item.find_elements(By.XPATH, ".//span[contains(@class,'hds-text-body-medium') and contains(@class,'text-gray-primary') and contains(@class,'line-clamp-3')]")

                        if len(good_spans) >= 1:

                            good_point = good_spans[0].text.strip()

                    except:

                        pass

            

            # 아쉬운점 찾기 (bad.svg 이미지가 있는 div 내의 span)

            bad_point = None

            try:

                # 방법 1: bad.svg 이미지가 있는 div를 찾고 그 안의 span 찾기 (가장 안정적)

                bad_div = review_item.find_element(By.XPATH, ".//div[contains(@class,'flex') and contains(@class,'items-start') and contains(@class,'gap-x-8')]//img[@src='/svgs/bad.svg']/ancestor::div[contains(@class,'flex') and contains(@class,'items-start')][1]")

                bad_spans = bad_div.find_elements(By.XPATH, ".//span[contains(@class,'hds-text-body-medium') and contains(@class,'text-gray-primary') and contains(@class,'line-clamp-3')]")

                if len(bad_spans) >= 1:

                    bad_point = bad_spans[0].text.strip()

            except:

                # 방법 2: 더 간단한 XPath로 시도

                try:

                    bad_div = review_item.find_element(By.XPATH, ".//img[@src='/svgs/bad.svg']/ancestor::div[contains(@class,'flex')][1]")

                    bad_spans = bad_div.find_elements(By.XPATH, ".//span[contains(@class,'line-clamp-3')]")

                    if len(bad_spans) >= 1:

                        bad_point = bad_spans[0].text.strip()

                except:

                    # 방법 3: line-clamp-3 span 중 두 번째 (fallback)

                    try:

                        bad_spans = review_item.find_elements(By.XPATH, ".//span[contains(@class,'hds-text-body-medium') and contains(@class,'text-gray-primary') and contains(@class,'line-clamp-3')]")

                        if len(bad_spans) >= 2:

                            bad_point = bad_spans[1].text.strip()

                    except:

                        pass

            

            # 리뷰 정보가 하나라도 있으면 추가 (닉네임은 필수, 좋은점 또는 아쉬운점 중 하나는 있어야 함)

            if nickname and (good_point or bad_point):

                review_data = {

                    "nickname": nickname,

                    "user_type": user_type,

                    "good_point": good_point,

                    "bad_point": bad_point,

                }

                reviews.append(review_data)

                print(f"  [DEBUG] Extracted review {len(reviews)}: {nickname} (good: {bool(good_point)}, bad: {bool(bad_point)})")

                

                if len(reviews) >= max_reviews:

                    break

            else:

                print(f"  [DEBUG] Skipping review {idx+1} (nickname: {nickname}): good={bool(good_point)}, bad={bool(bad_point)}")

                    

        except Exception as e:

            print(f"  [DEBUG] Error extracting review {idx+1}: {e}")

            continue

    

    # 위 방식이 실패하면 더 간단한 방법 시도

    if len(reviews) < max_reviews:

        try:

            # 닉네임으로 리뷰 블록 찾기

            nickname_elems = driver.find_elements(By.XPATH, "//span[contains(@class,'hds-text-subtitle-medium') and contains(@class,'text-gray-primary')]")

            
            for nickname_elem in nickname_elems[:max_reviews]:

                try:

                    # 닉네임 기준으로 부모 컨테이너 찾기

                    review_container = nickname_elem.find_element(By.XPATH, "./ancestor::div[position()<10]")

                    
                    nickname = nickname_elem.text.strip()

                    
                    # 사용자 타입

                    user_type = None

                    try:

                        type_elem = review_container.find_element(By.XPATH, ".//span[contains(@class,'hds-text-smalltext-large') and contains(@class,'text-gray-secondary')]")

                        user_type = type_elem.text.strip()

                    except:

                        pass

                    
                    # 좋아요 찾기 (line-clamp-3 span 중 첫 번째)

                    good_point = None

                    try:

                        good_spans = review_container.find_elements(By.XPATH, ".//span[contains(@class,'hds-text-body-medium') and contains(@class,'text-gray-primary') and contains(@class,'line-clamp-3')]")

                        if len(good_spans) >= 1:

                            good_point = good_spans[0].text.strip()

                    except:

                        pass

                    

                    # 이미지 기반 fallback

                    if not good_point:

                        try:

                            good_elem = review_container.find_element(By.XPATH, ".//img[@src='/svgs/good.svg']/following-sibling::span[contains(@class,'line-clamp-3')]")

                            good_point = good_elem.text.strip()

                        except:

                            pass

                    
                    # 아쉬워요 찾기 (line-clamp-3 span 중 두 번째)

                    bad_point = None

                    try:

                        bad_spans = review_container.find_elements(By.XPATH, ".//span[contains(@class,'hds-text-body-medium') and contains(@class,'text-gray-primary') and contains(@class,'line-clamp-3')]")

                        if len(bad_spans) >= 2:

                            bad_point = bad_spans[1].text.strip()

                    except:

                        pass

                    

                    # 이미지 기반 fallback

                    if not bad_point:

                        try:

                            bad_elem = review_container.find_element(By.XPATH, ".//img[@src='/svgs/bad.svg']/following-sibling::span[contains(@class,'line-clamp-3')]")

                            bad_point = bad_elem.text.strip()

                        except:

                            pass

                    
                    review_data = {

                        "nickname": nickname,

                        "user_type": user_type,

                        "good_point": good_point,

                        "bad_point": bad_point,

                    }

                    # 중복 체크

                    if review_data not in reviews:

                        reviews.append(review_data)

                        if len(reviews) >= max_reviews:

                            break

                except:

                    continue

        except:

            pass

    

    return reviews[:max_reviews]





# -----------------------------

# Step 1) Collect product list only (fast)

# -----------------------------

def collect_product_list_only(driver, subcat: str) -> List[Dict[str, str]]:
    """
    랭킹 페이지에서 제품 목록(제품명, 브랜드, URL)만 빠르게 수집
    상세 페이지로 이동하지 않음
    """
    print(f"\n[STEP 1] Collecting product list for '{subcat}'...")
    
    # 하위카테고리 선택 -> 랭킹 페이지로 이동
    select_subcategory_chip(driver, subcat)
    human_sleep(2.0, 3.0)
    
    # 제품 링크 수집 (상세 페이지로 이동하지 않고 목록만)
    products = collect_product_links(driver, limit=MAX_PRODUCTS_PER_SUBCAT)
    
    print(f"  [SUCCESS] Collected {len(products)} products for '{subcat}'")
    
    return products


def save_product_list(all_products: List[Dict[str, str]], subcategory: str):
    """제품 목록을 CSV 파일로 저장"""
    if not all_products:
        print(f"  [WARNING] No products to save for '{subcategory}'")
        return
    
    # 기존 파일이 있으면 읽어서 병합
    existing_products = []
    if os.path.exists(PRODUCT_LIST_FILE):
        try:
            df_existing = pd.read_csv(PRODUCT_LIST_FILE, encoding="utf-8-sig")
            existing_products = df_existing.to_dict('records')
            print(f"  [INFO] Found existing product list with {len(existing_products)} products")
        except:
            pass
    
    # 중복 제거 (URL 기준)
    existing_urls = {p.get("url") for p in existing_products if p.get("url")}
    new_products = [p for p in all_products if p.get("url") not in existing_urls]
    
    # 병합
    all_products_merged = existing_products + new_products
    
    # subcategory 추가
    for p in all_products_merged:
        if "subcategory" not in p:
            p["subcategory"] = subcategory
    
    # 저장
    df = pd.DataFrame(all_products_merged)
    df.to_csv(PRODUCT_LIST_FILE, index=False, encoding="utf-8-sig")
    print(f"  [SUCCESS] Saved {len(all_products_merged)} products to {PRODUCT_LIST_FILE} (added {len(new_products)} new products)")


# -----------------------------

# Step 2) Collect detailed information from saved list

# -----------------------------

def collect_product_details_from_list(driver):
    """
    저장된 제품 목록 파일을 읽어서 각 제품의 상세 정보 수집
    """
    if not os.path.exists(PRODUCT_LIST_FILE):
        print(f"[ERROR] Product list file not found: {PRODUCT_LIST_FILE}")
        print("[INFO] Please run with CRAWL_MODE='list_only' or 'both' first")
        return []
    
    # 제품 목록 읽기
    df_list = pd.read_csv(PRODUCT_LIST_FILE, encoding="utf-8-sig")
    products = df_list.to_dict('records')
    
    print(f"\n[STEP 2] Reading product list from {PRODUCT_LIST_FILE}")
    print(f"  [INFO] Found {len(products)} products to process")
    
    collected_rows = []
    
    for idx, product in enumerate(products, start=1):
        url = product.get("url")
        subcategory = product.get("subcategory", "")
        product_name_from_list = product.get("product_name")
        brand_from_list = product.get("brand")
        
        if not url:
            continue
        
        print(f"\n  [{idx}/{len(products)}] Processing: {product_name_from_list} ({brand_from_list})")
        print(f"    URL: {url}")
        
        try:
            # 상세로 새 탭 열기
            open_new_tab(driver, url)
            
            # 상세 정보 수집
            detail = extract_product_detail(driver)
            
            # 랭킹 페이지에서 가져온 제품명과 브랜드로 덮어쓰기
            if product_name_from_list:
                detail["product_name"] = product_name_from_list
            if brand_from_list:
                detail["brand"] = brand_from_list
            
            # 리뷰 샘플
            reviews = extract_review_snippets(driver, max_reviews=MAX_REVIEW_SNIPPETS)
            
            # 리뷰를 CSV에 저장 가능한 형태로 변환
            row = {
                "subcategory": subcategory,
                **detail,
            }
            
            # 각 리뷰를 별도 컬럼으로 추가
            for i, review in enumerate(reviews[:2], start=1):
                row[f"review_{i}_nickname"] = review.get("nickname")
                row[f"review_{i}_user_type"] = review.get("user_type")
                row[f"review_{i}_good_point"] = review.get("good_point")
                row[f"review_{i}_bad_point"] = review.get("bad_point")
            
            # 리뷰가 2개 미만인 경우 빈 값으로 채우기
            for i in range(len(reviews), 2):
                row[f"review_{i+1}_nickname"] = None
                row[f"review_{i+1}_user_type"] = None
                row[f"review_{i+1}_good_point"] = None
                row[f"review_{i+1}_bad_point"] = None
            
            collected_rows.append(row)
            
            # 탭 닫고 목록으로 복귀
            close_tab_back(driver)
            
            # 체크포인트 저장
            if len(collected_rows) % CHECKPOINT_EVERY == 0:
                try:
                    df_ckpt = pd.DataFrame(collected_rows)
                    df_ckpt.to_csv(OUTFILE, index=False, encoding="utf-8-sig")
                    print(f"      [checkpoint saved] rows={len(collected_rows)} -> {OUTFILE}")
                except PermissionError:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_file = OUTFILE.replace(".csv", f"_backup_{timestamp}.csv")
                    df_ckpt = pd.DataFrame(collected_rows)
                    df_ckpt.to_csv(backup_file, index=False, encoding="utf-8-sig")
                    print(f"      [WARNING] Original file locked, saved to backup: {backup_file}")
                except Exception as e:
                    print(f"      [ERROR] Failed to save checkpoint: {e}")
            
            # 제품 사이 대기 시간
            wait_time = random.uniform(2.0, 4.0)
            print(f"      [waiting {wait_time:.1f}s before next product...]")
            human_sleep(2.0, 4.0)
            
            # 가끔 더 긴 휴식
            if random.random() > 0.7:
                long_break = random.uniform(5.0, 10.0)
                print(f"      [taking a longer break: {long_break:.1f}s...]")
                human_sleep(5.0, 10.0)
        
        except Exception as e:
            print(f"      [ERROR] Failed to process product: {e}")
            import traceback
            traceback.print_exc()
            try:
                close_tab_back(driver)
            except:
                pass
            continue
    
    return collected_rows


# -----------------------------

# Main crawl

# -----------------------------

def crawl():

    driver = make_driver(headless=HEADLESS)

    collected_rows = []



    try:
        
        # 모드에 따라 분기
        if CRAWL_MODE == "details_only":
            # 저장된 제품 목록에서 상세 정보만 수집
            print("\n" + "="*60)
            print("[MODE] details_only - Collecting detailed information from saved list")
            print("="*60)
            
            driver.get(START_URL)
            human_sleep(2.0, 3.0)
            
            collected_rows = collect_product_details_from_list(driver)
            
            # 최종 저장
            if collected_rows:
                try:
                    df = pd.DataFrame(collected_rows)
                    df.to_csv(OUTFILE, index=False, encoding="utf-8-sig")
                    print(f"\n[done] total rows={len(collected_rows)} saved -> {OUTFILE}")
                except PermissionError:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_file = OUTFILE.replace(".csv", f"_backup_{timestamp}.csv")
                    df = pd.DataFrame(collected_rows)
                    df.to_csv(backup_file, index=False, encoding="utf-8-sig")
                    print(f"\n[WARNING] Original file locked, saved to backup: {backup_file}")
                    print(f"[done] total rows={len(collected_rows)} saved -> {backup_file}")
                except Exception as e:
                    print(f"\n[ERROR] Failed to save final CSV: {e}")
                    print(f"[INFO] Total rows collected: {len(collected_rows)}")
            
            return
        
        # list_only 또는 both 모드: 제품 목록 수집
        if CRAWL_MODE in ["list_only", "both"]:
            print("\n" + "="*60)
            print(f"[MODE] {CRAWL_MODE} - Collecting product list from rankings")
            print("="*60)
        
        driver.get(START_URL)

        # 초기 로딩 시간 증가 (봇 감지 회피)

        initial_wait = random.uniform(4.0, 6.0)

        print(f"[INFO] Page loaded, waiting {initial_wait:.1f}s for elements...")

        human_sleep(4.0, 6.0)

        

        # 초기 마우스 움직임 (사람처럼)

        random_mouse_movement(driver)

        

        # 카테고리 UI 진입 (순서 중요!)
        # 1단계: 카테고리별 탭 클릭 (급상승 탭에서 카테고리별 탭으로 이동)

        print("\n[STEP 1] Clicking '카테고리별' tab...")

        print(f"  [DEBUG] Current URL before clicking category tab: {driver.current_url}")

        

        try:

            click_category_tab(driver)

            

            # 탭 클릭 후 URL 확인

            final_url = driver.current_url

            print(f"  [DEBUG] Final URL after clicking category tab: {final_url}")

            

            if "english_name=category" in final_url or "theme_id=2" in final_url:

                print("  [SUCCESS] '카테고리별' tab clicked and navigated successfully")

            else:

                print(f"  [WARNING] '카테고리별' tab may not have worked. Current URL: {final_url}")

                print("  [WARNING] Continuing anyway, but may not be on category page...")

            

            human_sleep(2.0, 3.0)  # 탭 전환 후 충분한 대기 시간

            random_mouse_movement(driver)

        except Exception as e:

            print(f"[ERROR] Failed to click category tab: {e}")

            import traceback

            traceback.print_exc()

        

        # 2단계: 카테고리 전체 (검정 pill) 클릭 → 토글 열기

        print("\n[STEP 2] Clicking '카테고리 전체' (black pill) to open toggle...")

        try:

            click_black_category_all(driver)

            human_sleep(1.0, 1.5)  # 추가 대기

            random_mouse_movement(driver)

            print("  [SUCCESS] '카테고리 전체' clicked, toggle should be open")

        except Exception as e:

            print(f"[WARNING] Failed to click black category all: {e}")

        

        # 3단계: 스킨케어 클릭 (토글이 열린 후)

        print("\n[STEP 3] Clicking '스킨케어' in the toggle...")

        try:

            expand_skincare_in_panel(driver)

            human_sleep(1.5, 2.5)  # 스킨케어 클릭 후 하위 카테고리가 나타날 시간

            random_mouse_movement(driver)

            print("  [SUCCESS] '스킨케어' clicked, subcategories should be visible")

        except Exception as e:

            print(f"[WARNING] Failed to click skincare: {e}")



        # 크롤링할 카테고리 목록 결정

        categories_to_crawl = SELECTED_SUBCATEGORIES if SELECTED_SUBCATEGORIES else SUBCATEGORIES

        

        # 테스트 모드 설정

        products_per_subcat = 1 if TEST_MODE else MAX_PRODUCTS_PER_SUBCAT

        # 리뷰는 항상 2개 수집 (TEST_MODE와 관계없이)
        review_snippets = MAX_REVIEW_SNIPPETS  # 항상 2개

        

        if TEST_MODE:

            print("\n" + "="*60)

            print("[TEST MODE] Testing with 1 product only")

            print("="*60)

        

        if SELECTED_SUBCATEGORIES:

            print(f"\n[INFO] Selected categories to crawl: {SELECTED_SUBCATEGORIES}")

        else:

            print(f"\n[INFO] Crawling all categories: {SUBCATEGORIES}")

        

        for subcat in categories_to_crawl:

            print(f"\n[Subcategory] {subcat}")

            print(f"  [INFO] Current URL before selection: {driver.current_url}")



            # 카테고리별 탭이 선택되어 있는지 확인하고, 필요시 다시 클릭

            if not is_category_tab_selected(driver):

                print("  [INFO] '카테고리별' tab not selected, clicking again...")

                click_category_tab(driver)

                human_sleep(1.5, 2.5)

            

            # 하위카테고리 선택 -> 랭킹 페이지로 이동

            select_subcategory_chip(driver, subcat)

            

            # 하위 카테고리 선택 후에도 카테고리별 탭이 선택되어 있는지 확인

            human_sleep(1.0, 1.5)

            if not is_category_tab_selected(driver):

                print("  [WARNING] '카테고리별' tab was deselected, clicking again...")

                click_category_tab(driver)

                human_sleep(2.0, 3.0)

            

            # 랭킹 페이지인지 확인

            current_url = driver.current_url

            print(f"  [INFO] Current URL after selection: {current_url}")

            print(f"  [INFO] Category tab selected: {is_category_tab_selected(driver)}")

            

            # 제품 링크 수집 전에 카테고리별 탭이 선택되어 있는지 최종 확인

            if not is_category_tab_selected(driver):

                print("  [WARNING] '카테고리별' tab not selected before collecting links, fixing...")

                click_category_tab(driver)

                human_sleep(2.0, 3.0)

            

            # 목록에서 제품 링크 수집 (랭킹 순서대로)

            products = collect_product_links(driver, limit=products_per_subcat)

            print(f"  - collected products: {len(products)}")
            
            if len(products) == 0:
                print(f"  [WARNING] No products found for '{subcat}'. Skipping...")
                continue

            # list_only 모드: 제품 목록만 저장하고 다음 카테고리로
            if CRAWL_MODE == "list_only":
                save_product_list(products, subcat)
                print(f"  [SUCCESS] Saved product list for '{subcat}'")
                continue

            # both 모드: 제품 목록 저장 후 상세 정보 수집
            if CRAWL_MODE == "both":
                save_product_list(products, subcat)

            for idx, product in enumerate(products, start=1):

                url = product["url"]
                product_name_from_list = product.get("product_name")
                brand_from_list = product.get("brand")

                print(f"    [{idx}/{len(products)}] {url}")
                print(f"      Product: {product_name_from_list}, Brand: {brand_from_list}")



                # 상세로 새 탭 열기

                open_new_tab(driver, url)



                # 상세 정보 (제품명과 브랜드는 랭킹 페이지에서 가져온 것 사용)

                detail = extract_product_detail(driver)
                
                # 랭킹 페이지에서 가져온 제품명과 브랜드로 덮어쓰기
                if product_name_from_list:
                    detail["product_name"] = product_name_from_list
                if brand_from_list:
                    detail["brand"] = brand_from_list

                # 리뷰 샘플

                reviews = extract_review_snippets(driver, max_reviews=review_snippets)



                # 리뷰를 CSV에 저장 가능한 형태로 변환

                row = {

                    "subcategory": subcat,

                    **detail,

                }

                

                # 각 리뷰를 별도 컬럼으로 추가 (최대 2개)

                for i, review in enumerate(reviews[:2], start=1):

                    row[f"review_{i}_nickname"] = review.get("nickname")

                    row[f"review_{i}_user_type"] = review.get("user_type")

                    row[f"review_{i}_good_point"] = review.get("good_point")

                    row[f"review_{i}_bad_point"] = review.get("bad_point")

                

                # 리뷰가 2개 미만인 경우 빈 값으로 채우기

                for i in range(len(reviews), 2):

                    row[f"review_{i+1}_nickname"] = None

                    row[f"review_{i+1}_user_type"] = None

                    row[f"review_{i+1}_good_point"] = None

                    row[f"review_{i+1}_bad_point"] = None

                

                collected_rows.append(row)



                # 탭 닫고 목록으로 복귀

                close_tab_back(driver)



                # 체크포인트 저장

                if len(collected_rows) % CHECKPOINT_EVERY == 0:

                    try:

                        df_ckpt = pd.DataFrame(collected_rows)

                        df_ckpt.to_csv(OUTFILE, index=False, encoding="utf-8-sig")

                        print(f"      [checkpoint saved] rows={len(collected_rows)} -> {OUTFILE}")

                    except PermissionError:

                        # 파일이 열려있을 경우 타임스탬프 추가한 파일명으로 저장

                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                        backup_file = OUTFILE.replace(".csv", f"_backup_{timestamp}.csv")

                        df_ckpt = pd.DataFrame(collected_rows)

                        df_ckpt.to_csv(backup_file, index=False, encoding="utf-8-sig")

                        print(f"      [WARNING] Original file locked, saved to backup: {backup_file}")

                    except Exception as e:

                        print(f"      [ERROR] Failed to save checkpoint: {e}")



                # 제품 사이 더 긴 대기 시간 (봇 감지 회피)

                wait_time = random.uniform(2.0, 4.0)

                print(f"      [waiting {wait_time:.1f}s before next product...]")

                human_sleep(2.0, 4.0)

                

                # 가끔 더 긴 휴식 (사람처럼)

                if random.random() > 0.7:

                    long_break = random.uniform(5.0, 10.0)

                    print(f"      [taking a longer break: {long_break:.1f}s...]")

                    human_sleep(5.0, 10.0)



            # 카테고리마다 과도한 요청 방지 (더 긴 대기)

            category_break = random.uniform(3.0, 6.0)

            print(f"  [waiting {category_break:.1f}s before next category...]")

            human_sleep(3.0, 6.0)



        # 최종 저장

        try:

            df = pd.DataFrame(collected_rows)

            df.to_csv(OUTFILE, index=False, encoding="utf-8-sig")

            print(f"\n[done] total rows={len(collected_rows)} saved -> {OUTFILE}")

        except PermissionError:

            # 파일이 열려있을 경우 타임스탬프 추가한 파일명으로 저장

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            backup_file = OUTFILE.replace(".csv", f"_backup_{timestamp}.csv")

            df = pd.DataFrame(collected_rows)

            df.to_csv(backup_file, index=False, encoding="utf-8-sig")

            print(f"\n[WARNING] Original file locked, saved to backup: {backup_file}")

            print(f"[done] total rows={len(collected_rows)} saved -> {backup_file}")

        except Exception as e:

            print(f"\n[ERROR] Failed to save final CSV: {e}")

            print(f"[INFO] Total rows collected: {len(collected_rows)}")



    finally:

        driver.quit()





if __name__ == "__main__":

    crawl()

