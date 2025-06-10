from gtts import gTTS
import os

# 음성 합성
tts = gTTS("장갑을 착용했습니다.", lang='ko')

# 파일 저장
tts.save("/tmp/gloves.mp3")

# mp3 파일 재생
os.system("mpg321 /tmp/gloves.mp3")
