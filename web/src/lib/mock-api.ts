import { POST_FEATURES, type PostFeature } from "./feature-labels";
import type { ScreeningOutput } from "./schema";

/**
 * Mocked screening result. Mirrors the backend `SinglePredictionResponse`
 * shape (camelCase on this side; phase 5+ wires the real client and converts
 * snake_case → camelCase). `threshold` is included so the result card can
 * draw the marker without needing a separate config call — when we wire the
 * real API we'll either echo it from the backend or expose it via /health.
 */
export type ShapContribution = {
  feature: PostFeature;
  shapValue: number;
};

export type ShapExplanation = {
  baseValue: number;
  finalValue: number;
  contributions: ShapContribution[];
};

export type ScreeningResult = {
  probability: number;
  ciLower: number;
  ciUpper: number;
  isPositive: boolean;
  threshold: number;
  shap: ShapExplanation;
};

const THRESHOLD = 0.467;
const MOCK_DELAY_MS = 600;
// Approximate population-average log-odds for ~14% diabetes/prediabetes
// prevalence — ln(0.14 / 0.86) ≈ -1.81. Used as the SHAP base value.
const BASE_VALUE = -1.81;

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
const logit = (p: number) => Math.log(p / (1 - p));

/**
 * Heuristic stand-in for the trained model. The point isn't accuracy — it's
 * that the UI feels alive while we iterate on layout. A few high-signal
 * inputs nudge the probability up or down so different patient profiles
 * produce visibly different result cards. Determinism keeps Storybook-style
 * design review reproducible.
 */
function mockProbability(input: ScreeningOutput): number {
  let p = 0.18;

  // Age: each decade above 30 adds ~6%.
  if (typeof input.RIDAGEYR === "number") {
    p += Math.max(0, (input.RIDAGEYR - 30) / 10) * 0.06;
  }

  // Waist-to-height ratio: >0.55 is the established at-risk threshold.
  if (typeof input.BMXWAIST === "number" && typeof input.BMXHT === "number") {
    const ratio = input.BMXWAIST / input.BMXHT;
    if (ratio > 0.55) p += (ratio - 0.55) * 1.4;
  }

  if (input.told_high_bp === true) p += 0.08;
  if (input.told_high_cholesterol === true) p += 0.06;

  if (typeof input.systolic_bp === "number" && input.systolic_bp >= 140) {
    p += 0.05;
  }

  if (
    typeof input.vigorous_minutes_per_week === "number" &&
    input.vigorous_minutes_per_week >= 150
  ) {
    p -= 0.05;
  }

  if (typeof input.phq9_score === "number" && input.phq9_score >= 10) {
    p += 0.04;
  }

  return clamp01(p);
}

/**
 * Per-feature directional pull, in raw log-odds-ish units. Hand-tuned so
 * features the operator actually changed produce visibly different bars.
 * These get rescaled below so they sum to (finalValue - baseValue), which
 * keeps `final_value = base_value + sum(shap_value)` consistent with the
 * backend invariant.
 */
function rawContribution(
  feature: PostFeature,
  input: ScreeningOutput,
): number {
  switch (feature) {
    case "young_adult":
      return input.RIDAGEYR < 45 ? -0.45 : 0.05;
    case "elderly":
      return input.RIDAGEYR >= 80 ? 0.5 : -0.05;
    case "waist_to_height_ratio":
      if (input.BMXWAIST == null || input.BMXHT == null) return 0.0;
      return (input.BMXWAIST / input.BMXHT - 0.55) * 2.5;
    case "told_high_cholesterol_missing":
      return input.told_high_cholesterol == null ? 0.15 : -0.05;
    case "told_high_bp":
      if (input.told_high_bp == null) return 0.0;
      return input.told_high_bp ? 0.55 : -0.2;
    case "told_high_cholesterol":
      if (input.told_high_cholesterol == null) return 0.0;
      return input.told_high_cholesterol ? 0.4 : -0.15;
    case "is_female":
      return input.is_female ? -0.05 : 0.05;
    case "drinking_frequency":
      if (input.drinking_frequency == null) return 0.0;
      return (input.drinking_frequency - 2) * 0.05;
    case "diastolic_bp":
      if (input.diastolic_bp == null) return 0.0;
      return (input.diastolic_bp - 80) / 30;
    case "systolic_bp":
      if (input.systolic_bp == null) return 0.0;
      return (input.systolic_bp - 120) / 25;
    case "education_level":
      return -0.08 * (input.education_level - 2);
    case "phq9_score":
      if (input.phq9_score == null) return 0.0;
      return input.phq9_score * 0.04;
    case "vigorous_minutes_per_week":
      if (input.vigorous_minutes_per_week == null) return 0.0;
      return input.vigorous_minutes_per_week >= 150 ? -0.15 : 0.05;
  }
}

function buildShap(input: ScreeningOutput, probability: number): ShapExplanation {
  const finalValue = logit(Math.min(0.999, Math.max(0.001, probability)));
  const target = finalValue - BASE_VALUE;

  const raws = POST_FEATURES.map((f) => ({
    feature: f,
    raw: rawContribution(f, input),
  }));
  const rawSum = raws.reduce((acc, r) => acc + r.raw, 0);
  // If the raws happen to cancel out, fall back to using them unscaled — the
  // visual remains informative even if the sum-invariant slips.
  const scale = Math.abs(rawSum) < 0.05 ? 1 : target / rawSum;

  const contributions = raws
    .map(({ feature, raw }) => ({
      feature,
      shapValue: raw * scale,
    }))
    .sort((a, b) => Math.abs(b.shapValue) - Math.abs(a.shapValue));

  return {
    baseValue: BASE_VALUE,
    finalValue,
    contributions,
  };
}

export async function predict(
  input: ScreeningOutput,
): Promise<ScreeningResult> {
  await new Promise((resolve) => setTimeout(resolve, MOCK_DELAY_MS));

  const probability = mockProbability(input);
  // Synthesise a plausible 95% CI: ±8 pp clamped to [0,1]. Real bootstrap CIs
  // from the backend will replace this.
  const ciLower = clamp01(probability - 0.08);
  const ciUpper = clamp01(probability + 0.08);

  return {
    probability,
    ciLower,
    ciUpper,
    isPositive: probability > THRESHOLD,
    threshold: THRESHOLD,
    shap: buildShap(input, probability),
  };
}
