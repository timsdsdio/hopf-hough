/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: ['./app/**/*.{html,ts}'],
  theme: {
    extend: {
      colors: { ink: { 900: '#0d131f', 950: '#070a12' } },
    },
  },
  plugins: [],
};
