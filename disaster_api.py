import requests
import urllib3
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

url = "https://www.safetydata.go.kr/V2/api/DSSP-IF-00247"
# 환경 변수에서 API 키 로드
serviceKey = os.getenv("SAFETY_DATA_SERVICE_KEY")

# API 키 로드 확인
if not serviceKey:
    print("오류: .env 파일에서 SAFETY_DATA_SERVICE_KEY 를 찾을 수 없거나 설정되지 않았습니다.")
    exit() # 키가 없으면 스크립트 종료

params = {
    "serviceKey": serviceKey,
    "returnType": "json",
    "pageNo": "1",
    "numOfRows": "5",
    "crtDt": "20250410",  # 오늘 날짜 기준
}

print(f"Requesting data with Service Key starting with: {serviceKey[:5]}...") # 키 일부만 로깅

response = requests.get(url, params=params, verify=False)

print("Status Code:", response.status_code)

# 응답 상태 확인
if response.status_code == 200:
    data = response.json()
    print(data)

    # 데이터 구조 확인 후 파싱
    items = data.get("body", [])  # body 아래에 바로 리스트 있음
    for i, item in enumerate(items, 1):
        print(f"\n[{i}]")
        print("일련번호:", item.get("SN", "없음"))
        print("생성일시:", item.get("CRT_DT", "없음"))
        print("메시지내용:", item.get("MSG_CN", "없음"))
        print("수신지역명:", item.get("RCPTN_RGN_NM", "없음"))
        print("긴급단계명:", item.get("EMRG_STEP_NM", "없음"))
        print("재해구분명:", item.get("DST_SE_NM", "없음"))
        print("등록일자:", item.get("REG_YMD", "없음"))
        print("수정일자:", item.get("MDFCN_YMD", "없음"))
else:
    print("요청 실패:", response.status_code)
    print("응답 내용:", response.text)
