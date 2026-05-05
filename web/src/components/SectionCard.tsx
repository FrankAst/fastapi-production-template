import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

type SectionCardProps = {
  number: number;
  title: string;
  icon: LucideIcon;
  required?: boolean;
  children: ReactNode;
};

export function SectionCard({
  number,
  title,
  icon: Icon,
  required,
  children,
}: SectionCardProps) {
  return (
    <section className="rounded-card border border-border bg-surface p-6 shadow-card">
      <header className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="flex h-6 w-6 items-center justify-center rounded-md bg-primary-tint">
            <Icon className="h-3.5 w-3.5 text-primary" strokeWidth={2} />
          </span>
          <h2 className="text-base font-bold leading-[22px] text-text">
            {number}. {title}
          </h2>
        </div>
        {required ? (
          <span className="text-xs font-semibold text-primary">
            ● Requerido
          </span>
        ) : null}
      </header>
      <div className="mt-5">{children}</div>
    </section>
  );
}
