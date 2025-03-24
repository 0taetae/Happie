# 1️⃣ 빌드 단계 (Next.js 정적 사이트 빌드)
FROM node:22 AS build

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build
RUN npm run export  # 정적 파일 생성

# 2️⃣ Nginx 서버 단계
FROM nginx:latest

# 정적 파일을 Nginx의 루트 디렉토리로 복사
COPY --from=build /app/out /usr/share/nginx/html

# Nginx 설정 파일 복사
COPY ./nginx.conf /etc/nginx/conf.d/default.conf

# 포트 설정
EXPOSE 80
EXPOSE 443

CMD ["nginx", "-g", "daemon off;"]
