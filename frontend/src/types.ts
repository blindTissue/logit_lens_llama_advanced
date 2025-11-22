export interface TokenPrediction {
  token: string;
  prob: number;
  id: number;
}

export interface LayerData {
  layer_index: number;
  layer_name: string;
  predictions: TokenPrediction[];
}

export interface InferenceResponse {
  text: string;
  logit_lens: LayerData[];
}

export interface InterventionConfig {
  type: "scale" | "zero" | "add" | "block_attention";
  value?: number;
  vector?: number[];
  token_index?: number;
  source_tokens?: number[];
  target_tokens?: number[];
}

export interface Interventions {
  [hook_name: string]: InterventionConfig;
}
