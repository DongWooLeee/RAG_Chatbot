import requests
from bs4 import BeautifulSoup

def search_web_for_answer(query):
    """웹에서 관련 정보를 검색하여 상위 3개 결과 반환"""
    search_url = f"https://www.google.com/search?q={query}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(search_url, headers=headers)
        if response.status_code != 200:
            return "❌ 웹 검색 중 오류 발생: 검색 결과를 가져오지 못했습니다."
        
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        # Google 검색 결과에서 상위 3개의 결과 추출
        for g in soup.find_all("div", class_="tF2Cxc")[:3]:
            title = g.find("h3").text if g.find("h3") else "제목 없음"
            link = g.find("a")["href"] if g.find("a") else "#"
            snippet = g.find("span").text if g.find("span") else "요약 없음"
            results.append({"title": title, "link": link, "snippet": snippet})
        
        return results

    except Exception as e:
        return f"❌ 웹 검색 중 오류 발생: {str(e)}"
