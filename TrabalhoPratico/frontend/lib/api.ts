export type HealthResponse = {
  app_name: string;
  version: string;
  static_model_loaded: boolean;
  dynamic_model_loaded: boolean;
  model_dir: string;
  mediapipe_version: string;
};

export type StaticPredictionResponse = {
  letter: string | null;
  confidence: number;
  hand_detected: boolean;
  annotated_frame_b64?: string | null;
  backend: string;
};

export type DynamicPredictionResponse = {
  word: string | null;
  confidence: number;
  body_detected: boolean;
  top_predictions: Array<{ label: string; confidence: number }>;
  annotated_frame_b64?: string | null;
  backend: string;
};

export type PhraseCorrectionResponse = {
  original: string;
  corrected: string;
  strategy: string;
  metadata: Record<string, unknown>;
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = init?.headers
    ? {
        "Content-Type": "application/json",
        ...init.headers,
      }
    : { "Content-Type": "application/json" };

  const response = await fetch(`${API_URL}${path}`, {
    headers,
    ...init,
  });

  if (!response.ok) {
    throw new Error(`API request failed: ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthResponse>("/health"),
  predictStatic: (frame: string) =>
    request<StaticPredictionResponse>("/predict", {
      method: "POST",
      body: JSON.stringify({ frame }),
    }),
  predictDynamic: (frames: string[]) =>
    request<DynamicPredictionResponse>("/predict_dynamic", {
      method: "POST",
      body: JSON.stringify({ frames }),
    }).catch(() => null),
  correctPhrase: (phrase: string) =>
    request<PhraseCorrectionResponse>("/llm_correct", {
      method: "POST",
      body: JSON.stringify({ phrase }),
    }),
  speak: (text: string) =>
    request<{ success: boolean; text_spoken: string }>("/speak", {
      method: "POST",
      body: JSON.stringify({ text }),
    }),
};
