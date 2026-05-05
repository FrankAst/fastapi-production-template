import { useState } from "react";
import { FormProvider, useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { AppHeader } from "./components/AppHeader";
import { AppFooter } from "./components/AppFooter";
import { PatientDataSection } from "./components/sections/PatientDataSection";
import { MeasurementsSection } from "./components/sections/MeasurementsSection";
import { HistorySection } from "./components/sections/HistorySection";
import { LifestyleSection } from "./components/sections/LifestyleSection";
import { MentalHealthSection } from "./components/sections/MentalHealthSection";
import { ResultCard } from "./components/ResultCard";
import { ApiError, predict, type ScreeningResult } from "./lib/api";
import {
  SCREENING_DEFAULTS,
  screeningSchema,
  type ScreeningInput,
  type ScreeningOutput,
} from "./lib/schema";

export default function App() {
  const methods = useForm<ScreeningInput, unknown, ScreeningOutput>({
    resolver: zodResolver(screeningSchema),
    mode: "onTouched",
    defaultValues: SCREENING_DEFAULTS,
  });

  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<ScreeningResult | null>(null);
  const [submittedOutput, setSubmittedOutput] =
    useState<ScreeningOutput | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const handleValid = async (values: ScreeningOutput) => {
    setSubmitting(true);
    setSubmitError(null);
    try {
      const next = await predict(values);
      setResult(next);
      setSubmittedOutput(values);
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (err) {
      setSubmitError(formatSubmitError(err));
      window.scrollTo({ top: 0, behavior: "smooth" });
    } finally {
      setSubmitting(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setSubmittedOutput(null);
    setSubmitError(null);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleClear = () => {
    methods.reset(SCREENING_DEFAULTS);
    // Controller-bound boolean fields don't always reflect `undefined` from
    // reset() — re-set them explicitly so the SegmentedToggles return to the
    // "no selection" state seen on a fresh page load.
    methods.setValue("is_female", undefined as unknown as boolean);
    methods.setValue("told_high_bp", undefined);
    methods.setValue("told_high_cholesterol", undefined);
    setSubmitError(null);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <FormProvider {...methods}>
      <form
        onSubmit={methods.handleSubmit(handleValid)}
        className="min-h-screen bg-bg"
        noValidate
      >
        <AppHeader />

        <main className="mx-auto max-w-[1200px] px-4 pb-[88px] pt-6 sm:px-6">
          {submitError && (
            <div
              role="alert"
              className="mb-4 rounded-md border border-refer/40 bg-refer/10 px-4 py-3 text-sm text-text"
            >
              <p className="font-bold">No se pudo evaluar</p>
              <p className="mt-1 text-text-muted">{submitError}</p>
            </div>
          )}
          {result && submittedOutput ? (
            <ResultCard result={result} output={submittedOutput} />
          ) : (
            <div className="flex flex-col gap-4">
              <PatientDataSection />
              <MeasurementsSection />
              <HistorySection />
              <LifestyleSection />
              <MentalHealthSection />
            </div>
          )}
        </main>

        <AppFooter
          submitting={submitting}
          hasResult={result !== null}
          onReset={handleReset}
          onClear={handleClear}
        />
      </form>
    </FormProvider>
  );
}

function formatSubmitError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 0) {
      return "No se pudo contactar al servidor. Verifique su conexión e intente de nuevo.";
    }
    if (err.status === 422) {
      return "Algunos datos no superaron la validación del servidor. Revise los campos e intente de nuevo.";
    }
    if (err.status >= 500) {
      return "El servidor encontró un error. Intente nuevamente en unos minutos.";
    }
    return `Error inesperado del servidor (código ${err.status}).`;
  }
  return "Ocurrió un error inesperado. Intente de nuevo.";
}
