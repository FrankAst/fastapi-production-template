import { z } from "zod";

/**
 * Two-tier validation: presence is *optional* for everything except age, sex,
 * and education level. Range is *always* enforced on whatever the user types.
 *
 * Bounds mirror `src/app/ml_binaries/training_input_schema.yaml` so the form
 * rejects values the API would reject anyway. Cross-field rules mirror
 * `src/app/domain/schema_validator.py`.
 *
 * `null` means "answered as No sabe / unknown" — the API accepts null for the
 * nullable columns. `undefined` / `""` means "not touched yet."
 */

const COMMA_TO_DOT = /,/g;

function parseNumeric(val: unknown): unknown {
  if (val === "" || val === undefined || val === null) return null;
  if (typeof val === "string") {
    const trimmed = val.trim();
    if (trimmed === "") return null;
    const num = Number(trimmed.replace(COMMA_TO_DOT, "."));
    return Number.isNaN(num) ? val : num;
  }
  return val;
}

function parseRequiredNumeric(val: unknown): unknown {
  const parsed = parseNumeric(val);
  return parsed === null ? undefined : parsed;
}

const optionalNumber = (min: number, max: number, rangeMsg: string) =>
  z.preprocess(
    parseNumeric,
    z
      .number({ error: rangeMsg })
      .min(min, { error: rangeMsg })
      .max(max, { error: rangeMsg })
      .nullable(),
  );

const optionalInteger = (min: number, max: number, rangeMsg: string) =>
  z.preprocess(
    parseNumeric,
    z
      .number({ error: rangeMsg })
      .int({ error: rangeMsg })
      .min(min, { error: rangeMsg })
      .max(max, { error: rangeMsg })
      .nullable(),
  );

const requiredInteger = (
  min: number,
  max: number,
  msg: { required: string; range: string },
) =>
  z.preprocess(
    parseRequiredNumeric,
    z
      .number({ error: msg.required })
      .int({ error: msg.range })
      .min(min, { error: msg.range })
      .max(max, { error: msg.range }),
  );

export const screeningSchema = z
  .object({
    RIDAGEYR: requiredInteger(18, 120, {
      required: "Ingrese la edad.",
      range: "La edad debe estar entre 18 y 120 años.",
    }),
    is_female: z.boolean({ error: "Seleccione una opción." }),
    education_level: requiredInteger(0, 5, {
      required: "Seleccione una opción.",
      range: "Seleccione una opción válida.",
    }),

    BMXWAIST: optionalNumber(
      30,
      200,
      "La cintura debe estar entre 30 y 200 cm.",
    ),
    BMXHT: optionalNumber(60, 210, "La altura debe estar entre 60 y 210 cm."),
    systolic_bp: optionalNumber(
      50,
      260,
      "La presión sistólica debe estar entre 50 y 260 mmHg.",
    ),
    diastolic_bp: optionalNumber(
      20,
      160,
      "La presión diastólica debe estar entre 20 y 160 mmHg.",
    ),

    told_high_bp: z.boolean().nullish(),
    told_high_cholesterol: z.boolean().nullish(),

    drinking_frequency: optionalInteger(
      0,
      4,
      "Seleccione una opción válida.",
    ),
    vigorous_minutes_per_week: optionalNumber(
      0,
      2520,
      "Los minutos deben estar entre 0 y 2520 por semana.",
    ),

    phq9_score: optionalInteger(
      0,
      27,
      "El puntaje PHQ-9 debe estar entre 0 y 27.",
    ),
  })
  .refine(
    (data) =>
      data.systolic_bp == null ||
      data.diastolic_bp == null ||
      data.systolic_bp > data.diastolic_bp,
    {
      message: "La sistólica debe ser mayor que la diastólica.",
      path: ["systolic_bp"],
    },
  )
  .refine(
    (data) =>
      data.BMXWAIST == null ||
      data.BMXHT == null ||
      data.BMXWAIST < data.BMXHT,
    {
      message: "La cintura debe ser menor que la altura.",
      path: ["BMXWAIST"],
    },
  );

export type ScreeningInput = z.input<typeof screeningSchema>;
export type ScreeningOutput = z.output<typeof screeningSchema>;

export const SCREENING_DEFAULTS = {
  RIDAGEYR: "",
  is_female: undefined,
  education_level: "",
  BMXWAIST: "",
  BMXHT: "",
  systolic_bp: "",
  diastolic_bp: "",
  told_high_bp: undefined,
  told_high_cholesterol: undefined,
  drinking_frequency: "",
  vigorous_minutes_per_week: "",
  phq9_score: "",
} as unknown as ScreeningInput;

export const TOTAL_FIELDS = 12;

/** Counts fields the operator has interacted with — feeds the progress ring. */
export function countFilledFields(values: ScreeningInput): number {
  const isFilled = (v: unknown) =>
    v !== undefined && v !== null && v !== "";
  // Tri-state booleans: `null` means "No sabe" — that's an answer, count it.
  const isAnswered = (v: unknown) => v !== undefined;

  return [
    isFilled(values.RIDAGEYR),
    isFilled(values.is_female),
    isFilled(values.education_level),
    isFilled(values.BMXWAIST),
    isFilled(values.BMXHT),
    isFilled(values.systolic_bp),
    isFilled(values.diastolic_bp),
    isAnswered(values.told_high_bp),
    isAnswered(values.told_high_cholesterol),
    isFilled(values.drinking_frequency),
    isFilled(values.vigorous_minutes_per_week),
    isFilled(values.phq9_score),
  ].filter(Boolean).length;
}
