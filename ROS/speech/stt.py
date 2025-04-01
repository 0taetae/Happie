### 프론트엔드에서 전달받은 음성 파일을 텍스트로 변환 : STT ###

import whisper
import tempfile
import io
import numpy as np

# Whisper 모델 로드
# 모델을 한 번만 로드하고 재사용
model = whisper.load_model("small")

def transcribe_stt(audio_buffer):
    try:
        # BytesIO 데이터를 임시 파일로 저장하여 처리
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            temp_audio_file.write(audio_buffer.getvalue())  # BytesIO 데이터를 파일로 저장
            temp_audio_file.seek(0)  # 파일 포인터를 처음으로 이동

            # Whisper 모델을 통해 텍스트 변환
            result = model.transcribe(temp_audio_file.name, language="ko")

        return result["text"]
    except Exception as e:
        print(f"STT 처리 중 오류 발생: {e}")
        return None
