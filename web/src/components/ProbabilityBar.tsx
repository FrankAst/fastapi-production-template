import { cn } from "@/lib/utils";

type ProbabilityBarProps = {
  probability: number;
  ciLower: number;
  ciUpper: number;
  threshold: number;
  isPositive: boolean;
};

const toPercent = (v: number) => `${Math.round(v * 100)}%`;

const toThresholdLabel = (v: number) =>
  `Umbral ${v.toFixed(3).replace(".", ",")}`;

export function ProbabilityBar({
  probability,
  ciLower,
  ciUpper,
  threshold,
  isPositive,
}: ProbabilityBarProps) {
  const ciLeftPct = ciLower * 100;
  const ciWidthPct = Math.max(0, (ciUpper - ciLower) * 100);
  const thresholdPct = threshold * 100;
  const probabilityPct = probability * 100;

  const ariaLabel =
    `Probabilidad estimada ${toPercent(probability)}, ` +
    `intervalo de confianza ${toPercent(ciLower)} a ${toPercent(ciUpper)}, ` +
    `umbral de derivación ${toPercent(threshold)}.`;

  return (
    <div role="img" aria-label={ariaLabel} className="px-6 pb-3 pt-7">
      <div className="relative">
        <span
          className="absolute -top-5 -translate-x-1/2 whitespace-nowrap text-[11px] font-semibold leading-4 text-refer tabular-nums"
          style={{ left: `${thresholdPct}%` }}
        >
          {toThresholdLabel(threshold)}
        </span>

        <div className="relative h-1.5 w-full rounded-full bg-border">
          <div
            className={cn(
              "absolute top-1/2 h-3 -translate-y-1/2 rounded-sm",
              isPositive ? "bg-refer-bg" : "bg-ok-bg",
            )}
            style={{ left: `${ciLeftPct}%`, width: `${ciWidthPct}%` }}
            aria-hidden="true"
          />
          <div
            className={cn(
              "absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-surface",
              isPositive ? "bg-refer" : "bg-ok",
            )}
            style={{ left: `${probabilityPct}%` }}
            aria-hidden="true"
          />
          <div
            className="absolute top-1/2 h-4 w-[2px] -translate-x-1/2 -translate-y-1/2 bg-refer"
            style={{ left: `${thresholdPct}%` }}
            aria-hidden="true"
          />
        </div>

        <div className="mt-1.5 flex justify-between text-[11px] font-medium leading-4 text-text-subtle tabular-nums">
          <span>0%</span>
          <span>100%</span>
        </div>
      </div>
    </div>
  );
}
