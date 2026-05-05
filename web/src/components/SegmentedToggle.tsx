import { useId } from "react";
import { cn } from "@/lib/utils";

type SegmentedValue = string | number | boolean | null;

export type SegmentedOption<T extends SegmentedValue> = {
  value: T;
  label: string;
  /** Marks the "no sabe" option so screen readers know it sends as missing. */
  isUnknown?: boolean;
};

type SegmentedToggleProps<T extends SegmentedValue> = {
  label: string;
  options: SegmentedOption<T>[];
  value: T | undefined;
  onChange: (value: T) => void;
  helper?: string;
  error?: string;
};

export function SegmentedToggle<T extends SegmentedValue>({
  label,
  options,
  value,
  onChange,
  helper,
  error,
}: SegmentedToggleProps<T>) {
  const groupId = useId();
  const helperId = `${groupId}-helper`;
  const hasError = Boolean(error);

  return (
    <div className="flex flex-col gap-1">
      <span
        id={groupId}
        className="text-[13px] font-semibold leading-[18px] text-text-muted"
      >
        {label}
      </span>

      <div
        role="radiogroup"
        aria-labelledby={groupId}
        aria-describedby={helper || error ? helperId : undefined}
        className={cn(
          "flex h-9 overflow-hidden rounded-input border",
          hasError ? "border-error" : "border-border",
        )}
      >
        {options.map((option, index) => {
          const isSelected = value !== undefined && option.value === value;
          const isLast = index === options.length - 1;
          return (
            <button
              key={String(option.value)}
              type="button"
              role="radio"
              aria-checked={isSelected}
              aria-label={
                option.isUnknown
                  ? `${option.label} — se enviará como dato faltante`
                  : undefined
              }
              onClick={() => onChange(option.value)}
              className={cn(
                "flex-1 px-4 text-sm font-semibold transition-colors",
                "focus-visible:z-10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-tint",
                !isLast && "border-r",
                isSelected
                  ? "bg-primary text-primary-fg border-primary"
                  : cn(
                      "bg-surface text-text-muted hover:bg-surface-muted",
                      hasError ? "border-error" : "border-border",
                    ),
              )}
            >
              {option.label}
            </button>
          );
        })}
      </div>

      {error ? (
        <p
          id={helperId}
          className="text-xs font-medium leading-4 text-error"
        >
          {error}
        </p>
      ) : helper ? (
        <p
          id={helperId}
          className="text-xs font-medium leading-4 text-text-subtle"
        >
          {helper}
        </p>
      ) : null}
    </div>
  );
}
