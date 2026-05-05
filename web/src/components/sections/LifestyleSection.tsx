import { useFormContext } from "react-hook-form";
import { Activity } from "lucide-react";
import { SectionCard } from "../SectionCard";
import { Select } from "../Select";
import { TextInput } from "../TextInput";
import { DRINKING_FREQUENCY_OPTIONS } from "@/lib/enums";
import type { ScreeningInput } from "@/lib/schema";

export function LifestyleSection() {
  const {
    register,
    formState: { errors },
  } = useFormContext<ScreeningInput>();

  return (
    <SectionCard number={4} title="Estilo de vida" icon={Activity}>
      <div className="grid grid-cols-1 gap-x-6 gap-y-4 md:grid-cols-2">
        <Select
          label="Frecuencia de consumo de alcohol"
          placeholder="Seleccione una opción"
          options={[...DRINKING_FREQUENCY_OPTIONS]}
          error={errors.drinking_frequency?.message}
          {...register("drinking_frequency")}
        />

        <TextInput
          label="Minutos de actividad vigorosa por semana"
          type="number"
          inputMode="numeric"
          step={1}
          min={0}
          max={2520}
          placeholder="Ej.: 90"
          unit="min"
          error={errors.vigorous_minutes_per_week?.message}
          {...register("vigorous_minutes_per_week")}
        />
      </div>
    </SectionCard>
  );
}
