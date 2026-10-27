import nextVitals from "eslint-config-next/core-web-vitals";

export default [
  { ignores: [".next/**", "out/**", "build/**", "next-env.d.ts"] },
  ...nextVitals,
  {
    // Keep the new performance diagnostic visible without rewriting tested session handoffs.
    rules: { "react-hooks/set-state-in-effect": "warn" },
  },
];
