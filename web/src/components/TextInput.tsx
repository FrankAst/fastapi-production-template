import { forwardRef, useId, type InputHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

type TextInputProps = Omit<InputHTMLAttributes<HTMLInputElement>, "id"> & {
  label: string;
  /** Optional unit suffix (e.g. "cm", "mmHg"). */
  unit?: string;
  /** Helper text shown below when there is no error. */
  helper?: string;
  /** Inline error message. When set, the input renders in error state. */
  error?: string;
  /** Optional explicit id; otherwise generated. */
  id?: string;
};

export const TextInput = forwardRef<HTMLInputElement, TextInputProps>(
  (
    { label, unit, helper, error, id, className, ...rest },
    ref,
  ) => {
    const reactId = useId();
    const inputId = id ?? reactId;
    const helperId = `${inputId}-helper`;
    const hasError = Boolean(error);

    return (
      <div className="flex flex-col gap-1">
        <label
          htmlFor={inputId}
          className="text-[13px] font-semibold leading-[18px] text-text-muted"
        >
          {label}
        </label>

        <div className="relative">
          <input
            ref={ref}
            id={inputId}
            aria-invalid={hasError || undefined}
            aria-describedby={helper || error ? helperId : undefined}
            className={cn(
              "h-11 w-full rounded-input border bg-surface px-3.5 text-[15px] leading-[22px] text-text",
              "placeholder:text-text-subtle",
              "focus:outline-none focus:ring-2 focus:ring-primary-tint",
              hasError
                ? "border-error focus:border-error"
                : "border-border focus:border-primary",
              unit ? "pr-12" : undefined,
              className,
            )}
            {...rest}
          />
          {unit ? (
            <span className="pointer-events-none absolute right-3.5 top-1/2 -translate-y-1/2 text-[13px] font-medium text-text-subtle">
              {unit}
            </span>
          ) : null}
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
  },
);

TextInput.displayName = "TextInput";
