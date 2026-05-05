import { forwardRef, useId, type SelectHTMLAttributes } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

export type SelectOption = {
  value: string;
  label: string;
};

type SelectProps = Omit<SelectHTMLAttributes<HTMLSelectElement>, "id"> & {
  label: string;
  options: SelectOption[];
  /** Placeholder shown as a disabled first option when no value is selected. */
  placeholder?: string;
  helper?: string;
  error?: string;
  id?: string;
};

export const Select = forwardRef<HTMLSelectElement, SelectProps>(
  ({ label, options, placeholder, helper, error, id, className, ...rest }, ref) => {
    const reactId = useId();
    const selectId = id ?? reactId;
    const helperId = `${selectId}-helper`;
    const hasError = Boolean(error);

    // Stays uncontrolled by default so it works with react-hook-form's
    // `register`. If the consumer passes `value` (controlled, e.g. via
    // Controller) it flows through `...rest`. `defaultValue=""` makes the
    // disabled placeholder option the initial selection.
    const defaultValue = placeholder ? "" : undefined;

    return (
      <div className="flex flex-col gap-1">
        <label
          htmlFor={selectId}
          className="text-[13px] font-semibold leading-[18px] text-text-muted"
        >
          {label}
        </label>

        <div className="relative">
          <select
            ref={ref}
            id={selectId}
            defaultValue={defaultValue}
            aria-invalid={hasError || undefined}
            aria-describedby={helper || error ? helperId : undefined}
            className={cn(
              "h-11 w-full appearance-none rounded-input border bg-surface pl-3.5 pr-9 text-[15px] leading-[22px] text-text",
              "focus:outline-none focus:ring-2 focus:ring-primary-tint",
              "invalid:text-text-subtle",
              hasError
                ? "border-error focus:border-error"
                : "border-border focus:border-primary",
              className,
            )}
            {...rest}
          >
            {placeholder ? (
              <option value="" disabled>
                {placeholder}
              </option>
            ) : null}
            {options.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          <ChevronDown
            className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-text-muted"
            strokeWidth={2}
            aria-hidden="true"
          />
        </div>

        {error ? (
          <p id={helperId} className="text-xs font-medium leading-4 text-error">
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

Select.displayName = "Select";
