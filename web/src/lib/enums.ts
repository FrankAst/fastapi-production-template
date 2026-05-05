/**
 * Spanish (es-AR) labels for the IntEnums consumed by the API.
 *
 * Source of truth: `src/app/domain/enums.py` and the Feature Mapping section
 * of `docs/claude_code_guides/Frontend PR/frontend.md`.
 *
 * The numeric values must match the IntEnum members exactly — they ride the
 * wire to the prediction endpoint and are passed straight into the model.
 */

export const EDUCATION_LEVEL_OPTIONS = [
  { value: "0", label: "Prefiero no responder / No sabe" },
  { value: "1", label: "Menos de 9.º grado" },
  { value: "2", label: "Secundario incompleto" },
  { value: "3", label: "Secundario completo" },
  { value: "4", label: "Terciario / universitario incompleto" },
  { value: "5", label: "Universitario completo o más" },
] as const;

export const DRINKING_FREQUENCY_OPTIONS = [
  { value: "0", label: "Nunca" },
  { value: "1", label: "Ocasional" },
  { value: "2", label: "Mensual" },
  { value: "3", label: "Semanal" },
  { value: "4", label: "Diario" },
] as const;

const optionLookup = (
  options: readonly { value: string; label: string }[],
): Record<number, string> =>
  Object.fromEntries(options.map((o) => [Number(o.value), o.label]));

const EDUCATION_LEVEL_LABELS = optionLookup(EDUCATION_LEVEL_OPTIONS);
const DRINKING_FREQUENCY_LABELS = optionLookup(DRINKING_FREQUENCY_OPTIONS);

export const educationLevelLabel = (value: number): string | null =>
  EDUCATION_LEVEL_LABELS[value] ?? null;

export const drinkingFrequencyLabel = (value: number): string | null =>
  DRINKING_FREQUENCY_LABELS[value] ?? null;
