# Chapter1. 서비스 소개

## 1-1. 개요

|||
|:---:|:---|
|서비스 제품명|**하피(happie)**|
|프로젝트명|AI와 ROS2를 활용한 병원 도우미|
|도메인|모빌리티 > 스마트홈|
|기간|2025.03.04.(화) ~ 2025.04.11.(금) **6주**|

## 1-2. 기획 배경 및 목표

병원은 인간의 삶에서 결코 없어지 않는 공간으로, 항상 의료 인력이 부족하다.<br/>이를 해결하고자 병원 내 로봇을 도입하고자 한다.<br/>직원들의 업무 부담을 감소하고, 나아가 환자들에게 더 나은 환경을 제공한다.

## 1-3. 서비스 요약

**하피**는 AI와 ROS2를 활용한 병원 도우미 로봇이다.<br/>SSAFY에서 제공되는 시뮬레이터 환경에서 터틀봇이 이 역할을 수행한다.<br/>여러 알고리즘과 하피에게 탑재된 카메라, RiDA 등 다양한 센서를 이용해 실내 자율주행한다.<br/>관리자는 시스템을 통해 하피의 위치, 상태 등을 확인하고 명령을 지시할 수 있다.<br/>하피는 상시 원내를 순회하며 사람과 교류하거나 사고를 미연에 대비한다.<br/>대표적으로 쓰러진 사람을 인지하여 직원에게 알리며 사고에 즉각적인 대처를 요한다.<br/>또한, 하피는 실시간 대화형 챗봇으로 다양한 질의응답을 할 수 있다.

# Chapter2. 주요 기능

## 2-1. ROS2 자율주행

## 2-2. 기자재 관리 : 링거폴대 및 휠체어

## 2-3. 낙상사고 감지 및 대응

## 2-4. 대화형 음성인식 챗봇

음성 또는 자주 하는 질문을 선택하여 하피에게 물을 수 있다.<br/>텍스트 및 음성의 응답을 제공 받는다.<br/>병원 또는 원내 시설 관련, 이외 다양한 일상 대화를 나눈다.<br/>특정 장소로 하피에게 직접 주행 안내를 받을 수 있다.

![챗봇](/uploads/f427ec801737925e8b3519e525d32601/챗봇.gif)

# Chapter3. 시연 영상

# Chapter4. 기술

## RAG 기반 LLM 대화형 챗봇

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

# Chapter5. 산출물

## 5-1. 패키지 구조도

## 5-2. UI/UX 설계

### Figma

![image](/uploads/74d67ec564976be9e4c8c73a090d638c/image.png){width=552 height=608}

### 시뮬레이션

![스크린샷_2025-03-20_134020](/uploads/02727e6896fd7ee71d305d16ff33b97b/스크린샷_2025-03-20_134020.png){width=800}

## 5-3. DB 설계

![image](/uploads/3d95604a84002090cd9beea420710f26/image.png){width=748 height=497}

## 5-4. 아키텍처 구조

![최종_구조도](/uploads/691d1df4fb784a89f963ad0ee9f8c180/최종_구조도.png)

## 5-5. API 및 MQTT 통신 설계

# Chatper6. 팀 구성

![image](/uploads/3387e093c5adbfd0d0eee3c5cd772269/image.png){width=794 height=464}
