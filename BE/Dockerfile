# Docker 이미지의 베이스 이미지를 설정
FROM openjdk:17-jdk

# 작업 디렉토리 설정
WORKDIR /spring-boot

# 호스트 시스템의 target 디렉토리에 있는 JAR 파일을 컨테이너의 /spring-boot/ 디렉토리로 복사
COPY target/foodthink-0.0.1-SNAPSHOT.jar /spring-boot/app.jar

COPY src/main/resources/application.properties /spring-boot/application.properties

# Spring Boot 애플리케이션 실행
ENTRYPOINT ["java", "-jar", "/spring-boot/app.jar"]
