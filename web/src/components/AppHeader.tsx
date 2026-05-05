import { Leaf } from "lucide-react";

export function AppHeader() {
  return (
    <header className="sticky top-0 z-20 h-16 w-full border-b border-border bg-surface">
      <div className="mx-auto flex h-full max-w-[1200px] items-center justify-between px-4 sm:px-6">
        <div className="flex items-center gap-2">
          <Leaf className="h-5 w-5 text-primary" strokeWidth={2} />
          <span className="text-lg font-bold leading-6 text-primary">
            Tamizaje Metabólico
          </span>
        </div>

        <div className="hidden flex-col items-end leading-tight sm:flex">
          <span className="text-sm font-semibold text-text">
            Farmacia San Martín
          </span>
          <span className="text-xs font-medium text-text-muted">
            Operadora: Mariana Pérez
          </span>
        </div>
      </div>
    </header>
  );
}
