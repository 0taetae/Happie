/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,ts,jsx,tsx}",
    "./src/app/globals.css" // Tailwind가 CSS 파일도 감지하도록 추가
  ],
  theme: {
    extend: {},
  },
  plugins: [],
};
