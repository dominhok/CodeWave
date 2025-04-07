from twilio.rest import Client

# api 키와 토큰 저장
account_sid = 'API_KEY'
auth_token = 'API_TOKEN'
client = Client(account_sid, auth_token)

# 데이터베이스나 데모 시연용으로 저희 전화번호나 해커톤 참가자들의 번호를 저장하면 될거 같아요
numbers_to_message = ['번호1', '번호2', '번호3', '...']

# 리스트의 모든 번호들에게 동일하게 전송
for number in numbers_to_message:
    message = client.messages.create(
        body='해커톤 연습용 알림 문자입니다.',
        from_='+12318036691',
        to=number
    )
    print(message.sid)
    print(message.body)
