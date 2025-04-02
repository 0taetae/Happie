import openai
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.llms import OpenAI

# OpenAI API 키 설정
openai.api_key = 'your-openai-api-key'  # 여기에 실제 API 키를 넣으세요.

class ConversationMemoryManager:
    def __init__(self):
        # 대화 이력을 기억할 메모리 객체 생성
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # OpenAI GPT 모델 객체 생성
        self.llm = OpenAI(model="gpt-3.5-turbo", openai_api_key=openai.api_key)  # GPT-3.5 모델 예시
        self.conversation = ConversationChain(memory=self.memory, llm=self.llm)

    def get_previous_conversations(self):
        # 대화 이력 반환
        return self.memory.buffer

    def add_to_memory(self, user_input, bot_output):
        # 사용자 입력과 봇 출력을 메모리에 추가
        self.memory.add_user_message(user_input)
        self.memory.add_ai_message(bot_output)

    def get_response(self, query):
        # 모델을 사용하여 응답 생성
        return self.conversation.predict(input=query)
