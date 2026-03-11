/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: "#22d3ee",
        secondary: "#a855f7",
        dark: "#080b14",
        "dark-card": "rgba(15, 23, 42, 0.6)",
        danger: "#fb7185",
        success: "#34d399",
        dim: "#94a3b8",
      },
      animation: {
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        glow: {
          'from': { boxShadow: '0 0 10px rgba(34, 211, 238, 0.2)' },
          'to': { boxShadow: '0 0 30px rgba(34, 211, 238, 0.5)' },
        }
      }
    },
  },
  plugins: [],
}
