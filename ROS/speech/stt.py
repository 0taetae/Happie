### 프론트엔드에서 전달받은 음성 파일을 텍스트로 변환 : STT ###
### (임시 구현) MQTT 통신으로 받은 값을 처리하도록 + 텍스트 출력값을 프롬프팅으로 전달 예정 ###

import whisper

# Whisper 모델 로드
def load_whisper():
    model = whisper.load_model("base")
    return model

# STT
def transcribe_stt(audio_file_path):
    # 모델 로드
    model = load_whisper()

    # 음성 파일을 텍스트로 변환
    print(f"음성 파일 '{audio_file_path}'을(를) 처리 중...")
    result = model.transcribe(audio_file_path)
    
    # 변환된 텍스트 출력
    print("변환된 텍스트:")
    print(result["text"])

    return result["text"]


if __name__ == "__main__":
    # 예시로 사용할 음성 파일과 출력 텍스트 파일 경로 설정
    audio_file = "다운로드\\샘플_2.wav"  # 여기서 'your_audio_file.wav'는 변환할 음성 파일입니다.

    # 음성 파일을 텍스트로 변환하고 그 결과를 변수로 받기
    # 나중에 이 텍스트를 프롬프트로 사용할 수 있음
    transcribed_text = transcribe_stt(audio_file)