import { ClipboardCheck, Eraser, RotateCcw } from "lucide-react";
import { useFormContext, useWatch } from "react-hook-form";
import { Button } from "./Button";
import { ProgressRing } from "./ProgressRing";
import {
  countFilledFields,
  TOTAL_FIELDS,
  type ScreeningInput,
} from "@/lib/schema";

type AppFooterProps = {
  submitting: boolean;
  hasResult: boolean;
  onReset: () => void;
  onClear: () => void;
};

export function AppFooter({
  submitting,
  hasResult,
  onReset,
  onClear,
}: AppFooterProps) {
  const {
    formState: { errors },
  } = useFormContext<ScreeningInput>();
  const values = useWatch<ScreeningInput>() as ScreeningInput;

  const completed = countFilledFields(values);
  const progress = completed / TOTAL_FIELDS;

  const requiredFilled =
    values.RIDAGEYR !== "" &&
    values.RIDAGEYR != null &&
    values.is_female !== undefined &&
    values.education_level !== "" &&
    values.education_level != null;
  const hasVisibleErrors = Object.keys(errors).length > 0;
  const canSubmit = requiredFilled && !hasVisibleErrors && !submitting;

  const helper = hasResult
    ? "Puede revisar los datos ingresados antes de imprimir."
    : requiredFilled
      ? "Puede revisar los datos antes de evaluar."
      : "Complete los campos requeridos para evaluar.";

  return (
    <footer className="fixed inset-x-0 bottom-0 z-20 h-[72px] border-t border-border bg-surface">
      <div className="mx-auto flex h-full max-w-[1200px] items-center justify-between gap-4 px-4 sm:px-6">
        <div className="flex items-center gap-3">
          <ProgressRing progress={progress} />
          <div className="flex flex-col leading-tight">
            <span className="text-sm font-bold text-text tabular-nums">
              {completed} de {TOTAL_FIELDS} campos completados
            </span>
            <span className="text-xs font-medium text-text-muted">
              {helper}
            </span>
          </div>
        </div>

        {hasResult ? (
          <Button
            key="reset"
            type="button"
            variant="primary"
            leadingIcon={<RotateCcw className="h-4 w-4" strokeWidth={2} />}
            onClick={(e) => {
              // React 18 batches state updates inside event handlers and
              // flushes them synchronously before the click's activation
              // behavior runs. Without preventDefault, the DOM button's
              // `type` flips to "submit" mid-event and the form submits.
              // The `key` swap below removes the same risk by mounting a
              // fresh DOM node, but preventDefault stays as a belt.
              e.preventDefault();
              onReset();
            }}
          >
            Evaluar de nuevo
          </Button>
        ) : (
          <div className="flex items-center gap-2">
            <Button
              key="clear"
              type="button"
              variant="secondary"
              leadingIcon={<Eraser className="h-4 w-4" strokeWidth={2} />}
              onClick={(e) => {
                e.preventDefault();
                onClear();
              }}
              disabled={completed === 0 || submitting}
            >
              Limpiar
            </Button>
            <Button
              key="submit"
              type="submit"
              variant="primary"
              leadingIcon={
                <ClipboardCheck className="h-4 w-4" strokeWidth={2} />
              }
              disabled={!canSubmit}
            >
              {submitting ? "Evaluando…" : "Evaluar"}
            </Button>
          </div>
        )}
      </div>
    </footer>
  );
}
