import { AlertTriangle, Check } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ScreeningResult } from "@/lib/mock-api";
import type { ScreeningOutput } from "@/lib/schema";
import { ProbabilityBar } from "./ProbabilityBar";
import { ShapWaterfall } from "./ShapWaterfall";

type ResultCardProps = {
  result: ScreeningResult;
  output: ScreeningOutput;
};

const toPercent = (v: number) => `${Math.round(v * 100)}%`;

export function ResultCard({ result, output }: ResultCardProps) {
  const { probability, ciLower, ciUpper, isPositive, threshold, shap } = result;
  const Icon = isPositive ? AlertTriangle : Check;
  const label = isPositive
    ? "Derivar a evaluación"
    : "Sin indicación de derivación";

  return (
    <div className="flex flex-col items-center gap-4">
      <section
        className={cn(
          "w-full rounded-card border p-8",
          isPositive
            ? "border-refer-bg bg-refer-halo"
            : "border-ok-bg bg-ok-halo",
        )}
      >
        <div className="flex flex-col items-center gap-3">
          <span
            className={cn(
              "flex h-9 w-9 items-center justify-center rounded-full",
              isPositive ? "bg-refer-bg" : "bg-ok-bg",
            )}
          >
            <Icon
              className={cn(
                "h-5 w-5",
                isPositive ? "text-refer" : "text-ok",
              )}
              strokeWidth={2.25}
              aria-hidden="true"
            />
          </span>
          <span
            className={cn(
              "w-4/5 rounded-pill px-4 py-3 text-center text-lg font-bold leading-6",
              isPositive
                ? "bg-refer-bg text-refer"
                : "bg-ok-bg text-ok",
            )}
          >
            {label}
          </span>
        </div>

        <p className="mt-6 text-center tabular-nums">
          <span className="text-[40px] font-extrabold leading-[48px] text-text">
            {toPercent(probability)}
          </span>{" "}
          <span className="text-base font-medium leading-6 text-text-muted">
            (IC 95%: {toPercent(ciLower)}–{toPercent(ciUpper)})
          </span>
        </p>

        <ProbabilityBar
          probability={probability}
          ciLower={ciLower}
          ciUpper={ciUpper}
          threshold={threshold}
          isPositive={isPositive}
        />
      </section>

      <p className="text-center text-xs font-medium leading-4 text-text-muted">
        Herramienta de tamizaje, no diagnóstico.
      </p>

      <div className="w-full">
        <ShapWaterfall shap={shap} output={output} />
      </div>
    </div>
  );
}
