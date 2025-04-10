from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.twiml.voice_response import VoiceResponse, Gather

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'This is a testing app for SMS, Calls from Twilio!'

# 문자용 라우터터
@app.route("/sms", methods=['POST'])
def sms_reply():
    # 수신된 메세지 정보 msg에 저장
    msg = request.form.get('Body')
    print(f"\n**\n📩 받은 메시지: {msg}\n**\n")
    # 응답 로그 출력
    resp = MessagingResponse()
    resp.message(f"응답: '{msg}' 잘 받았어요!")
    return str(resp)

@app.route("/voice", methods=["POST"])
def voice():
    response = VoiceResponse()

    # 사용자 음성을 받기 위한 Gather 설정
    gather = Gather(
        input="speech",
        timeout=5,
        speechTimeout="auto",
        action="/gather",
        method="POST",
        language="ko-KR"
    )
    gather.say("안녕하세요. 음성을 말씀해 주세요.", voice='alice', language='ko-KR')
    response.append(gather)

    # 사용자가 말하지 않은 경우에도 종료
    response.hangup()
    return str(response)

@app.route("/gather", methods=["POST"])
def gather():
    speech_result = request.form.get("SpeechResult")
    print(f"\n**\n📢 사용자 음성 인식 결과: {speech_result}\n**\n")

    # 통화 종료 (Twilio가 자동으로 끊음, 응답 필요 없음)
    response = VoiceResponse()
    response.hangup()
    return str(response)

if __name__ == "__main__":
    app.run(debug=True)