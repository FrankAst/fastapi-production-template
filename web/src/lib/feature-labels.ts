import { drinkingFrequencyLabel, educationLevelLabel } from "./enums";
import type { ScreeningOutput } from "./schema";

/**
 * The 13 post-processed features the model actually consumes, in the order
 * `ColumnSelector` produces them. Names mirror
 * `src/app/services/processing/column_selector/config.py` so a SHAP payload
 * from the real backend lines up 1:1 with the keys here.
 */
export const POST_FEATURES = [
  "young_adult",
  "elderly",
  "waist_to_height_ratio",
  "told_high_cholesterol_missing",
  "told_high_bp",
  "told_high_cholesterol",
  "is_female",
  "drinking_frequency",
  "diastolic_bp",
  "systolic_bp",
  "education_level",
  "phq9_score",
  "vigorous_minutes_per_week",
] as const;

export type PostFeature = (typeof POST_FEATURES)[number];

/** Canonical Spanish labels — the waterfall renderer must go through this. */
export const FEATURE_LABELS: Record<PostFeature, string> = {
  young_adult: "Adulto joven (18–44 años)",
  elderly: "Adulto mayor (≥ 80 años)",
  waist_to_height_ratio: "Relación cintura/altura",
  told_high_cholesterol_missing: "Antecedente de colesterol no reportado",
  told_high_bp: "Antecedente de presión alta",
  told_high_cholesterol: "Antecedente de colesterol alto",
  is_female: "Sexo (femenino)",
  drinking_frequency: "Frecuencia de consumo de alcohol",
  diastolic_bp: "Presión diastólica",
  systolic_bp: "Presión sistólica",
  education_level: "Nivel educativo",
  phq9_score: "Puntaje PHQ-9 (depresión)",
  vigorous_minutes_per_week: "Minutos de actividad vigorosa por semana",
};

const numberFormatter1 = new Intl.NumberFormat("es-AR", {
  maximumFractionDigits: 1,
});
const numberFormatter2 = new Intl.NumberFormat("es-AR", {
  maximumFractionDigits: 2,
});

const yesNo = (v: boolean | null | undefined) =>
  v === true ? "Sí" : v === false ? "No" : "No sabe";

/**
 * Patient input value to bake into the bar label, e.g.
 * "Relación cintura/altura: 0,62". Returns null when the feature has no
 * meaningful operator-facing value (e.g. dummy flags) so the renderer can
 * decide whether to show a colon.
 */
export function formatFeatureValue(
  feature: PostFeature,
  output: ScreeningOutput,
): string | null {
  switch (feature) {
    case "young_adult":
      return output.RIDAGEYR < 45 ? "Sí" : "No";
    case "elderly":
      return output.RIDAGEYR >= 80 ? "Sí" : "No";

    case "waist_to_height_ratio": {
      if (output.BMXWAIST == null || output.BMXHT == null) return null;
      const ratio = output.BMXWAIST / output.BMXHT;
      return numberFormatter2.format(ratio);
    }

    case "told_high_cholesterol_missing":
      return output.told_high_cholesterol == null ? "Sí" : "No";
    case "told_high_bp":
      return yesNo(output.told_high_bp);
    case "told_high_cholesterol":
      return yesNo(output.told_high_cholesterol);

    case "is_female":
      return output.is_female ? "Sí" : "No";

    case "drinking_frequency":
      return output.drinking_frequency == null
        ? null
        : drinkingFrequencyLabel(output.drinking_frequency);

    case "diastolic_bp":
      return output.diastolic_bp == null
        ? null
        : `${numberFormatter1.format(output.diastolic_bp)} mmHg`;
    case "systolic_bp":
      return output.systolic_bp == null
        ? null
        : `${numberFormatter1.format(output.systolic_bp)} mmHg`;

    case "education_level":
      return educationLevelLabel(output.education_level);

    case "phq9_score":
      return output.phq9_score == null ? null : String(output.phq9_score);

    case "vigorous_minutes_per_week":
      return output.vigorous_minutes_per_week == null
        ? null
        : `${output.vigorous_minutes_per_week} min`;
  }
}

/**
 * True when the feature's source raw input was blank at submit-time, so the
 * model is reasoning from an imputed value and the UI should mark the bar
 * with the "(estimado)" suffix.
 *
 * The derived missingness flag (`told_high_cholesterol_missing`) is itself
 * never imputed — it captures the missingness honestly — so it's always
 * treated as observed. Required fields (age, sex, education) likewise can
 * never be imputed by construction.
 */
export function isImputedFromBlank(
  feature: PostFeature,
  output: ScreeningOutput,
): boolean {
  switch (feature) {
    case "young_adult":
    case "elderly":
    case "is_female":
    case "education_level":
    case "told_high_cholesterol_missing":
      return false;

    case "waist_to_height_ratio":
      return output.BMXWAIST == null || output.BMXHT == null;

    case "told_high_bp":
      return output.told_high_bp == null;
    case "told_high_cholesterol":
      return output.told_high_cholesterol == null;

    case "drinking_frequency":
      return output.drinking_frequency == null;
    case "diastolic_bp":
      return output.diastolic_bp == null;
    case "systolic_bp":
      return output.systolic_bp == null;
    case "phq9_score":
      return output.phq9_score == null;
    case "vigorous_minutes_per_week":
      return output.vigorous_minutes_per_week == null;
  }
}
