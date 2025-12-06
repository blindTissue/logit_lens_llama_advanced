export interface TokenPrediction {
  token: string;
  prob: number;
  id: number;
}

export interface LayerData {
  layer_index: number;
  layer_name: string;
  predictions: TokenPrediction[][]; // Array of token positions, each with top-k predictions
}

export interface InferenceResponse {
  text: string;
  input_tokens: string[]; // Actual input tokens for display
  logit_lens: LayerData[];
  attention?: number[][][][]; // [layers, heads, seq, seq]
}

export interface InterventionConfig {
  type: "scale" | "zero" | "add" | "block_attention";
  value?: number;
  vector?: number[];
  token_index?: number;
  source_tokens?: number[];
  target_tokens?: number[];
  all_layers?: boolean;
}

export type Interventions = Record<string, InterventionConfig[]>;
