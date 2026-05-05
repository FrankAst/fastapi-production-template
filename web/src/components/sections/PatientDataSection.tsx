import { Controller, useFormContext } from "react-hook-form";
import { User } from "lucide-react";
import { SectionCard } from "../SectionCard";
import { TextInput } from "../TextInput";
import { SegmentedToggle, type SegmentedOption } from "../SegmentedToggle";
import { Select } from "../Select";
import { EDUCATION_LEVEL_OPTIONS } from "@/lib/enums";
import type { ScreeningInput } from "@/lib/schema";

const SEX_OPTIONS: SegmentedOption<boolean>[] = [
  { value: false, label: "Masculino" },
  { value: true, label: "Femenino" },
];

export function PatientDataSection() {
  const {
    register,
    control,
    formState: { errors },
  } = useFormContext<ScreeningInput>();

  return (
    <SectionCard number={1} title="Datos del paciente" icon={User} required>
      <div className="grid grid-cols-1 gap-x-6 gap-y-4 md:grid-cols-3">
        <TextInput
          label="Edad"
          type="number"
          inputMode="numeric"
          min={18}
          max={120}
          step={1}
          placeholder="Ej.: 52"
          unit="años"
          error={errors.RIDAGEYR?.message}
          {...register("RIDAGEYR")}
        />

        <Controller
          control={control}
          name="is_female"
          render={({ field, fieldState }) => (
            <SegmentedToggle<boolean>
              label="Sexo"
              options={SEX_OPTIONS}
              value={field.value}
              onChange={field.onChange}
              error={fieldState.error?.message}
            />
          )}
        />

        <Select
          label="Nivel educativo"
          placeholder="Seleccione una opción"
          options={[...EDUCATION_LEVEL_OPTIONS]}
          error={errors.education_level?.message}
          {...register("education_level")}
        />
      </div>
    </SectionCard>
  );
}
