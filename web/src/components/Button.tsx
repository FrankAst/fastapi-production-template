import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from "react";
import { cn } from "@/lib/utils";

type ButtonVariant = "primary" | "secondary" | "ghost";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
  leadingIcon?: ReactNode;
};

const base =
  "inline-flex items-center justify-center gap-2 rounded-input px-5 min-h-[44px] " +
  "text-sm font-bold leading-5 transition-colors " +
  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-tint " +
  "focus-visible:border-primary disabled:cursor-not-allowed";

const variants: Record<ButtonVariant, string> = {
  primary:
    "bg-primary text-primary-fg hover:bg-primary-hover " +
    "disabled:bg-border-strong disabled:text-text-subtle",
  secondary:
    "bg-surface text-primary border border-border hover:bg-surface-muted " +
    "disabled:text-text-subtle",
  ghost:
    "bg-transparent text-primary hover:bg-primary-tint " +
    "disabled:text-text-subtle",
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ variant = "primary", leadingIcon, className, children, ...rest }, ref) => {
    return (
      <button
        ref={ref}
        className={cn(base, variants[variant], className)}
        {...rest}
      >
        {leadingIcon ? (
          <span className="flex h-4 w-4 items-center justify-center">
            {leadingIcon}
          </span>
        ) : null}
        <span>{children}</span>
      </button>
    );
  },
);

Button.displayName = "Button";
