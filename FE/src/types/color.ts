// src/types/color.ts

export const colorOptions = ["amber", "rose", "sky", "lime", "violet"] as const;

export type ColorType = (typeof colorOptions)[number];

export const colorClassMap: Record<ColorType, string> = {
  amber: "bg-amber-100 border-amber-300 hover:bg-amber-200",
  sky: "bg-sky-100 border-sky-300 hover:bg-sky-200",
  rose: "bg-rose-100 border-rose-300 hover:bg-rose-200",
  lime: "bg-lime-100 border-lime-300 hover:bg-lime-200",
  violet: "bg-violet-100 border-violet-300 hover:bg-violet-200",
};
