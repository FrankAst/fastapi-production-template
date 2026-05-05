import { Controller, useFormContext } from "react-hook-form";
import { HeartPulse } from "lucide-react";
import { SectionCard } from "../SectionCard";
import { SegmentedToggle, type SegmentedOption } from "../SegmentedToggle";
import type { ScreeningInput } from "@/lib/schema";

const HISTORY_OPTIONS: SegmentedOption<boolean | null>[] = [
  { value: true, label: "Sí" },
  { value: false, label: "No" },
  { value: null, label: "No sabe", isUnknown: true },
];

export function HistorySection() {
  const { control } = useFormContext<ScreeningInput>();

  return (
    <SectionCard number={3} title="Antecedentes" icon={HeartPulse}>
      <div className="grid grid-cols-1 gap-y-4">
        <Controller
          control={control}
          name="told_high_bp"
          render={({ field, fieldState }) => (
            <SegmentedToggle<boolean | null>
              label="¿Le han dicho alguna vez que tiene la presión alta?"
              options={HISTORY_OPTIONS}
              value={field.value === undefined ? undefined : field.value}
              onChange={field.onChange}
              error={fieldState.error?.message}
            />
          )}
        />

        <Controller
          control={control}
          name="told_high_cholesterol"
          render={({ field, fieldState }) => (
            <SegmentedToggle<boolean | null>
              label="¿Le han dicho alguna vez que tiene el colesterol alto?"
              options={HISTORY_OPTIONS}
              value={field.value === undefined ? undefined : field.value}
              onChange={field.onChange}
              error={fieldState.error?.message}
            />
          )}
        />
      </div>
    </SectionCard>
  );
}
