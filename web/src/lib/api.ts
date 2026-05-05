import { POST_FEATURES, type PostFeature } from "./feature-labels";
import type { ScreeningResult, ShapExplanation } from "./mock-api";
import type { ScreeningOutput } from "./schema";

export type { ScreeningResult } from "./mock-api";

/**
 * Real client for `POST /prediction/single`. Mirrors the mock's `predict()`
 * signature so swapping at the call site is a one-line import change.
 *
 * The backend's `BaseSchema` sets `alias_generator=to_camel`, and FastAPI's
 * default `response_model_by_alias=True` means responses are already in
 * camelCase — no boundary mapping needed. We only filter unknown SHAP
 * features and tack on the hardcoded threshold (see
 * `docs/claude_code_guides/Frontend PR/api-integration.md` step 5 for how
 * to expose it from the backend).
 */

const BASE = import.meta.env.VITE_API_BASE_URL ?? "/api";
const THRESHOLD = 0.467;

type RawShapContribution = { feature: string; shapValue: number };
type RawResponse = {
  probability: number;
  ciLower: number;
  ciUpper: number;
  isPositive: boolean;
  shap: {
    baseValue: number;
    finalValue: number;
    contributions: RawShapContribution[];
  };
};

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly detail: unknown,
  ) {
    super(`API error ${status}`);
    this.name = "ApiError";
  }
}

const KNOWN_FEATURES = new Set<string>(POST_FEATURES);
const isPostFeature = (s: string): s is PostFeature => KNOWN_FEATURES.has(s);

function toShap(raw: RawResponse["shap"]): ShapExplanation {
  return {
    baseValue: raw.baseValue,
    finalValue: raw.finalValue,
    contributions: raw.contributions
      .filter((c) => isPostFeature(c.feature))
      .map((c) => ({
        feature: c.feature as PostFeature,
        shapValue: c.shapValue,
      })),
  };
}

export async function predict(
  input: ScreeningOutput,
): Promise<ScreeningResult> {
  let res: Response;
  try {
    res = await fetch(`${BASE}/prediction/single`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    });
  } catch (err) {
    throw new ApiError(0, err instanceof Error ? err.message : String(err));
  }

  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new ApiError(res.status, detail);
  }

  const raw = (await res.json()) as RawResponse;
  return {
    probability: raw.probability,
    ciLower: raw.ciLower,
    ciUpper: raw.ciUpper,
    isPositive: raw.isPositive,
    threshold: THRESHOLD,
    shap: toShap(raw.shap),
  };
}
