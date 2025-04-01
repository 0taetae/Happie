/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,ts,jsx,tsx}",      // 모든 JS/TS/React 파일
    "./src/app/globals.css"            // 커스텀 CSS도 포함
  ],
  safelist: [
    "cursor-default-custom",           // 커스텀 커서 보호
    "cursor-pointer-custom"
  ],
  theme: {
    extend: {},
  },
  plugins: [],
};
