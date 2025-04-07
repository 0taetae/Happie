/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,ts,jsx,tsx}",
    "./src/app/globals.css"
  ],
  safelist: [
    "cursor-default-custom",
    "cursor-pointer-custom"
  ],
  theme: {
    extend: {
      fontFamily: {
        godo: ["'Godo'", "sans-serif"],
        bmjua: ["'BMJUA'", "sans-serif"],

      },
    },
  },
  plugins: [],
};
