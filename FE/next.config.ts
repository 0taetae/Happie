import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'export',
  images: {
    unoptimized: true, // 정적 사이트에서 이미지 최적화 기능 끔
  },
};

export default nextConfig;
