import { useState } from "react";
import { ChevronDown, ChevronUp, Info } from "lucide-react";
import { Button } from "./Button";
import { cn } from "@/lib/utils";
import {
  FEATURE_LABELS,
  formatFeatureValue,
  isImputedFromBlank,
  type PostFeature,
} from "@/lib/feature-labels";
import type { ShapExplanation } from "@/lib/mock-api";
import type { ScreeningOutput } from "@/lib/schema";

type ShapWaterfallProps = {
  shap: ShapExplanation;
  output: ScreeningOutput;
};

const DEFAULT_VISIBLE = 6;
// Longest visible bar fills 95% of its half-width per design-guidelines.
const MAX_BAR_PERCENT = 47.5;

export function ShapWaterfall({ shap, output }: ShapWaterfallProps) {
  const [expanded, setExpanded] = useState(false);
  // Drop the cholesterol-missingness flag when the operator actually answered
  // the cholesterol question — its non-zero SHAP comes from being below the
  // population-average missingness rate, which is mathematically valid but
  // confusing in the UI ("you didn't report cholesterol" right next to a
  // patient who clearly did).
  const contributions = shap.contributions.filter(
    (c) =>
      !(
        c.feature === "told_high_cholesterol_missing" &&
        output.told_high_cholesterol != null
      ),
  );
  const total = contributions.length;
  const visible = expanded
    ? contributions
    : contributions.slice(0, DEFAULT_VISIBLE);
  const maxAbs = visible.reduce(
    (acc, c) => Math.max(acc, Math.abs(c.shapValue)),
    0,
  );

  return (
    <section className="rounded-card border border-border bg-surface p-6 shadow-card">
      <header className="flex items-start gap-3">
        <span className="mt-0.5 flex h-6 w-6 items-center justify-center rounded-md bg-primary-tint">
          <Info className="h-3.5 w-3.5 text-primary" strokeWidth={2} />
        </span>
        <div>
          <h2 className="text-base font-bold leading-[22px] text-text">
            ¿Por qué este resultado?
          </h2>
          <p className="mt-1 text-[13px] leading-[18px] text-text-muted">
            Las principales contribuciones del modelo para este paciente.
          </p>
        </div>
      </header>

      <div className="mt-5 grid grid-cols-1 gap-x-4 sm:grid-cols-[minmax(160px,260px)_1fr]">
        <div className="hidden sm:block" aria-hidden="true" />
        <div className="flex justify-between pb-2 text-xs font-semibold">
          <span className="text-ok-bar">← Disminuye el riesgo</span>
          <span className="text-refer-bar">Aumenta el riesgo →</span>
        </div>
      </div>

      <ul role="list" className="flex flex-col">
        {visible.map((c) => (
          <ShapRow
            key={c.feature}
            feature={c.feature}
            shapValue={c.shapValue}
            maxAbs={maxAbs}
            output={output}
          />
        ))}
      </ul>

      {total > DEFAULT_VISIBLE ? (
        <div className="mt-3 flex justify-center">
          <Button
            type="button"
            variant="ghost"
            leadingIcon={
              expanded ? (
                <ChevronUp className="h-4 w-4" strokeWidth={2} />
              ) : (
                <ChevronDown className="h-4 w-4" strokeWidth={2} />
              )
            }
            onClick={() => setExpanded((v) => !v)}
            aria-expanded={expanded}
          >
            {expanded ? "Ver menos" : `Ver más (${total - DEFAULT_VISIBLE})`}
          </Button>
        </div>
      ) : null}

      <p className="mt-4 text-center text-xs font-medium leading-4 text-text-muted">
        Las contribuciones explican el cálculo del modelo, no son consejo clínico.
      </p>
    </section>
  );
}

type ShapRowProps = {
  feature: PostFeature;
  shapValue: number;
  maxAbs: number;
  output: ScreeningOutput;
};

function ShapRow({ feature, shapValue, maxAbs, output }: ShapRowProps) {
  const value = formatFeatureValue(feature, output);
  const imputed = isImputedFromBlank(feature, output);
  const isNegative = shapValue < 0;
  const widthPct =
    maxAbs > 0 ? (Math.abs(shapValue) / maxAbs) * MAX_BAR_PERCENT : 0;

  return (
    <li className="grid grid-cols-1 items-center gap-x-4 border-t border-border/40 py-2 first:border-t-0 sm:grid-cols-[minmax(160px,260px)_1fr]">
      <div className="text-[13px] leading-[18px] text-text">
        <span className="font-semibold">{FEATURE_LABELS[feature]}</span>
        {value ? (
          <span className="text-text-muted">
            {": "}
            {value}
          </span>
        ) : null}
        {imputed ? (
          <span className="text-text-subtle"> (estimado)</span>
        ) : null}
      </div>

      <div className="relative h-8">
        <div
          className="absolute inset-y-1 left-1/2 w-px bg-border"
          aria-hidden="true"
        />
        <div
          className={cn(
            "absolute top-1/2 h-3 -translate-y-1/2 rounded-sm",
            isNegative
              ? "right-1/2 origin-right bg-ok-bar"
              : "left-1/2 origin-left bg-refer-bar",
          )}
          style={{ width: `${widthPct}%` }}
          aria-hidden="true"
        />
      </div>
    </li>
  );
}
