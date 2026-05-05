import { useFormContext } from "react-hook-form";
import { Brain } from "lucide-react";
import { SectionCard } from "../SectionCard";
import { TextInput } from "../TextInput";
import type { ScreeningInput } from "@/lib/schema";

export function MentalHealthSection() {
  const {
    register,
    formState: { errors },
  } = useFormContext<ScreeningInput>();

  return (
    <SectionCard number={5} title="Salud mental" icon={Brain}>
      <div className="max-w-md">
        <TextInput
          label="Puntaje PHQ-9 (depresión)"
          type="number"
          inputMode="numeric"
          step={1}
          min={0}
          max={27}
          placeholder="Ej.: 4"
          helper="Suma de los 9 ítems del cuestionario PHQ-9 (0–27)."
          error={errors.phq9_score?.message}
          {...register("phq9_score")}
        />
      </div>
    </SectionCard>
  );
}
