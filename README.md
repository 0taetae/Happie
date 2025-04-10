# Chapter1. 서비스 소개

### 1-1. 개요

|||
|:---:|:---|
|서비스 제품명|**하피(happie)**|
|프로젝트명|AI와 ROS2를 활용한 병원 도우미 서비스스|
|도메인|모빌리티 > 스마트홈|
|기간|2025.03.04.(화) ~ 2025.04.11.(금) **6주**|

### 1-2. 기획 배경 및 목표

병원은 인간의 삶에서 결코 없어지 않는 공간으로, 항상 의료 인력이 부족하다. 이를 해결하고자 병원 내 로봇을 도입하고자 한다. 직원들의 업무 부담을 감소하고, 나아가 환자들에게 더 나은 환경을 제공한다.

### 1-3. 서비스 요약

**하피**는 AI와 ROS2를 활용한 병원 도우미 로봇이다. SSAFY에서 제공되는 시뮬레이터 환경에서 터틀봇이 이 역할을 수행한다. 여러 알고리즘과 하피에게 탑재된 카메라, 2D LiDAR 등 다양한 센서를 이용해 실내 자율주행한다.

관리자는 시스템을 통해 하피의 위치, 상태 등을 확인하고 명령을 지시할 수 있다. 하피는 상시 원내를 순회하며 사람과 교류하고, 사고에 대응할 수 있도록 한다. 대표적으로 쓰러진 사람을 인지하면 즉각적으로 반응해, 직원에게 알리고 사고에 대처를 요한다. 이 기능는 특히 사람이 적은 야간에 집중하여 병원 직원의 업무 부담을 감소하고 환자 안전을 보전한다. 또한, 하피는 실시간 대화형 챗봇으로 다양한 질의응답을 할 수 있다.

# Chapter2. 주요 기능

### 2-1. ROS2 자율주행

### 2-2. 기자재 관리 : 링거폴대 및 휠체어

### 2-3. 낙상사고 감지 및 대응

### 2-4. 대화형 음성인식 챗봇

음성 또는 자주 하는 질문을 선택하여 하피에게 물을 수 있다.<br/>텍스트 및 음성의 응답을 제공 받는다.<br/>병원 또는 원내 시설 관련, 이외 다양한 일상 대화를 나눈다.<br/>특정 장소로 하피에게 직접 주행 안내를 받을 수 있다.

![llm__online-video-cutter.com_](/uploads/a7c7e35d95b743780e711d3d6967b1ef/llm__online-video-cutter.com_.gif)

# Chapter3. 기술

### 2D LiDAR와 SLAM을 사용하여 실내 지도 생성

2D LiDAR 센서를 활용해 실내 물체까지의 거리 정보를 스캔한다. 이에 SLAM 알고리즘을 적용하여 실시간 지도 생성을 수행한다.

![매핑_10배속__online-video-cutter.com_](/uploads/331938db0f3dd298dd5593f16e429278/매핑_10배속__online-video-cutter.com_.gif)

### A* 알고리즘 기반 경로 계획

A* 알고리즘이란 시작 지점에서 목표 지점까지의 최적 경로를 찾는 알고리즘이다. 하피는 벽에 가까울수록 비용이 증가하며, 지고 내 안전 마진 5 이내의 비용 가중치를 부여해 최적 경로를 산출한다.

![a_star_result](/uploads/8edefab5f3421551812e29de8ba4a100/a_star_result.png)

### YOLOv5를 활용한 물체 인식 및 감지

시뮬레이터와 버전 문제와 클래스 단위의 분류가 가능한 특징을 활용하고자 YOLOv5를 사용하여 데이터셋을 학습한다. 정확도를 높이기 위해 데이터의 다양성과 수량을 증가한다. 이러한 데이터 학습은 기자재 관리와 낙상 감지 기능에 활용된다.

![_](/uploads/7bbdb75f4995bf709322c1a94ef37db6/_.png)

![2](/uploads/b1bc6aefc7b818f9787885313fd334ec/2.png)

### RAG 기반 LangChain 및 LLM 활용

RAG(Retrieval-Augmented Generation)란 외부 지식을 검색하고, 그 결과를 바탕으로 답변을 생성하는 방식이다. 특정 병원 배경의 대화가 중심으로, 정보의 정확성과 다양한 병원에서의 맞춤 정보 제공의 확장성을 기대하며 이 방법론을 사용한다. 엑셀 데이터를 OpenAI의 text embedding ada 002 임베딩 모델을 활용해 벡터화하여 ChromaDB에 사전에 저장 후, 이를 활용한다. LangChain을 활용해 효율적인 RAG 구조를 구성하며 사용자의 질문에 따른 응답을 제공한다.

1. 사용자 질문 이해
2. ChatGPT 3.5 turbo 활용 프롬프트 템플릿 기반 검색 질의 분석
3. RAG 기반 외부 정보 검색
4. ChatGPT 3.5 turbo 활용 프롬프트 템프릿 기반 응답 생성 및 정제 (대화 유도, 오류 대응)
5. 응답 산출

![image](/uploads/137fcb518a6f8acbe327c328bb1cb0bc/image.png){width=497 height=219}

### 기술 스택

![ROS2](https://img.shields.io/badge/ROS2-Robot%20OS-blueviolet)
![React](https://img.shields.io/badge/React-18.x-61DAFB?logo=react&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-13.x-000000?logo=next.js&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.x-38B2AC?logo=tailwindcss&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-4.x-3178C6?logo=typescript&logoColor=white)
![Java](https://img.shields.io/badge/Java-17-orange?logo=java&logoColor=white)
![Spring Boot](https://img.shields.io/badge/SpringBoot-3.x-6DB33F?logo=springboot&logoColor=white)
![JPA](https://img.shields.io/badge/JPA-Hibernate-59666C?logo=hibernate&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-8.x-4479A1?logo=mysql&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-purple)
![MQTT](https://img.shields.io/badge/MQTT-Mosquitto-FF8200?logo=eclipsemosquitto&logoColor=white)
![REST API](https://img.shields.io/badge/REST--API-REST-blue?logo=apachespark&logoColor=white)
![VSCode](https://img.shields.io/badge/VSCode-Editor-007ACC?logo=visualstudiocode&logoColor=white)
![IntelliJ](https://img.shields.io/badge/IntelliJ_IDEA-IDE-000000?logo=intellijidea&logoColor=white)
![GitLab](https://img.shields.io/badge/GitLab-CI/CD-FC6D26?logo=gitlab&logoColor=white)
![Jira](https://img.shields.io/badge/Jira-Tracking-0052CC?logo=jira&logoColor=white)
![Notion](https://img.shields.io/badge/Notion-Workspace-000000?logo=notion&logoColor=white)

# Chapter4. 산출물

### 4-1. 패키지 구조도

(유빈아 여기)

### 4-2. UI/UX 설계

#### Figma

![image](/uploads/74d67ec564976be9e4c8c73a090d638c/image.png){width=552 height=608}

#### 시뮬레이션

![스크린샷_2025-03-20_134020](/uploads/02727e6896fd7ee71d305d16ff33b97b/스크린샷_2025-03-20_134020.png){width=800}

### 4-3. DB 설계

![image](/uploads/3d95604a84002090cd9beea420710f26/image.png){width=748 height=497}

### 4-4. 아키텍처 구조

![최종_구조도](/uploads/691d1df4fb784a89f963ad0ee9f8c180/최종_구조도.png)

### 4-5. API 및 MQTT 통신 설계

(유빈아 여기)

# Chatper5. 팀 구성

![image](/uploads/3387e093c5adbfd0d0eee3c5cd772269/image.png){width=794 height=464}
