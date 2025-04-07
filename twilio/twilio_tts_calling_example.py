import os
from twilio.rest import Client

# api 키와 토큰 저장
account_sid = 'API_KEY'
auth_token = 'API_KEY'
client = Client(account_sid, auth_token)

call = client.calls.create(
    twiml="<Response><Say voice='alice' language='ko-KR'>안녕하세요. 해커톤 알림 전화입니다.</Say></Response>",
    to="+821042847030",
    from_='+12318036691',
)

print(call.sid)


"""
#####################
# 여러명에게 비동기적으로 TTS 전화를 거는 방식으로 동작하는 코드

import threading
from twilio.rest import Client

# Twilio 인증 정보
account_sid = 'API_KEY'
auth_token = 'API_KEY'
client = Client(account_sid, auth_token)

# 전화번호 리스트
phone_numbers = [
    '+821042847030',
    '+번호2',
    '+번호3',
    '+...'
]

# 전화를 거는 함수
def make_call(number):
    try:
        call = client.calls.create(
            twiml="<Response><Say voice='alice' language='ko-KR'>안녕하세요. 해커톤 알림 전화입니다.</Say></Response>",
            to=number,
            from_='+12318036691'
        )
        print(f"Call sent to {number} | SID: {call.sid}")
    except Exception as e:
        print(f"Failed to call {number} | Error: {e}")

# 각 번호에 대해 병렬로 실행
for number in phone_numbers:
    thread = threading.Thread(target=make_call, args=(number,))
    thread.start()
"""
