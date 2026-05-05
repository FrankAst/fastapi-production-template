import type { Config } from "tailwindcss";

/**
 * Tailwind theme — sourced from docs/claude_code_guides/Frontend PR/design-guidelines.md.
 *
 * Token values live as CSS variables in src/index.css so they can be referenced
 * from non-Tailwind CSS too. This config maps them onto Tailwind utilities so
 * components can write `bg-bg`, `text-text-muted`, `rounded-card`, `shadow-card`,
 * etc. without ever hand-typing a hex code.
 */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "var(--color-bg)",
        surface: "var(--color-surface)",
        "surface-muted": "var(--color-surface-muted)",
        border: "var(--color-border)",
        "border-strong": "var(--color-border-strong)",

        primary: {
          DEFAULT: "var(--color-primary)",
          hover: "var(--color-primary-hover)",
          tint: "var(--color-primary-tint)",
          fg: "var(--color-primary-fg)",
        },

        refer: {
          DEFAULT: "var(--color-refer)",
          bg: "var(--color-refer-bg)",
          halo: "var(--color-refer-halo)",
          bar: "var(--color-refer-bar)",
        },

        ok: {
          DEFAULT: "var(--color-ok)",
          bg: "var(--color-ok-bg)",
          halo: "var(--color-ok-halo)",
          bar: "var(--color-ok-bar)",
        },

        error: {
          DEFAULT: "var(--color-error)",
          bg: "var(--color-error-bg)",
        },

        text: {
          DEFAULT: "var(--color-text)",
          muted: "var(--color-text-muted)",
          subtle: "var(--color-text-subtle)",
        },
      },
      fontFamily: {
        sans: ["var(--font-sans)"],
        numeric: ["var(--font-numeric)"],
      },
      borderRadius: {
        card: "var(--radius-card)",
        input: "var(--radius-input)",
        pill: "var(--radius-pill)",
      },
      boxShadow: {
        card: "var(--shadow-card)",
        "card-hover": "var(--shadow-card-hover)",
      },
    },
  },
  plugins: [],
} satisfies Config;
