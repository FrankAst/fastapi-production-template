import { useFormContext } from "react-hook-form";
import { Pencil } from "lucide-react";
import { SectionCard } from "../SectionCard";
import { TextInput } from "../TextInput";
import type { ScreeningInput } from "@/lib/schema";

export function MeasurementsSection() {
  const {
    register,
    formState: { errors },
  } = useFormContext<ScreeningInput>();

  return (
    <SectionCard number={2} title="Mediciones" icon={Pencil}>
      <div className="grid grid-cols-1 gap-x-6 gap-y-4 md:grid-cols-2">
        <TextInput
          label="Cintura"
          type="number"
          inputMode="decimal"
          step="0.1"
          min={30}
          max={200}
          placeholder="Ej.: 92"
          unit="cm"
          error={errors.BMXWAIST?.message}
          {...register("BMXWAIST")}
        />

        <TextInput
          label="Altura"
          type="number"
          inputMode="decimal"
          step="0.1"
          min={60}
          max={210}
          placeholder="Ej.: 168"
          unit="cm"
          error={errors.BMXHT?.message}
          {...register("BMXHT")}
        />

        <TextInput
          label="Presión sistólica"
          type="number"
          inputMode="numeric"
          step={1}
          min={50}
          max={260}
          placeholder="Ej.: 128"
          unit="mmHg"
          error={errors.systolic_bp?.message}
          {...register("systolic_bp")}
        />

        <TextInput
          label="Presión diastólica"
          type="number"
          inputMode="numeric"
          step={1}
          min={20}
          max={160}
          placeholder="Ej.: 82"
          unit="mmHg"
          error={errors.diastolic_bp?.message}
          {...register("diastolic_bp")}
        />
      </div>
    </SectionCard>
  );
}
