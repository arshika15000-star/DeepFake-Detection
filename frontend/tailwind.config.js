/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: "#7ec8a0",      // Pistachio Green
        secondary: "#4ecdc4",    // Seafoam Mint
        dark: "#080d0a",
        "dark-card": "rgba(10, 20, 15, 0.6)",
        danger: "#fb7185",
        success: "#34d399",
        dim: "#94a3b8",
        light: {
          bg: "#f0faf4",
          card: "#ffffff",
          text: "#1a2e22",
          dim: "#4a7060",
        }
      },
      animation: {
        'glow': 'glow 2s ease-in-out infinite alternate',
        'float': 'float 6s ease-in-out infinite',
        'pulse-slow': 'pulse 4s ease-in-out infinite',
      },
      keyframes: {
        glow: {
          'from': { boxShadow: '0 0 10px rgba(126, 200, 160, 0.2)' },
          'to': { boxShadow: '0 0 30px rgba(126, 200, 160, 0.6)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-12px)' },
        }
      }
    },
  },
  plugins: [],
}
