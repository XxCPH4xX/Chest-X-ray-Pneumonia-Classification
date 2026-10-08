export type Label = "NORMAL" | "PNEUMONIA" | "INCONCLUSIVE";
export interface Prediction {
  prediction_id: string;
  created_at: string;
  predicted_class: Label;
  probabilities: { NORMAL: number; PNEUMONIA: number };
  threshold: number;
  architecture: string;
  model_version: string;
  gradcam_available: boolean;
  gradcam_target: string;
  uncertainty_policy: { low: number; high: number; source: string } | null;
  seconds: number;
  saved: boolean;
  images?: { model_input: string; heatmap: string; overlay: string } | null;
}
export interface ModelInfo {
  available: boolean;
  architecture: string;
  device: string;
  version: string | null;
  threshold: number | null;
  preprocessing: { image_size: number; resize: string } | null;
  uncertainty_policy: { low: number; high: number; source: string } | null;
  dataset: string;
  research_only: boolean;
  history_enabled: boolean;
}
export interface HistoryList {
  enabled: boolean;
  total: number;
  items: Prediction[];
}
export interface Metrics {
  accuracy: number | null;
  sensitivity: number | null;
  specificity: number | null;
  precision: number | null;
  f1: number | null;
  roc_auc: number | null;
  confusion_matrix: number[][];
  support: { NORMAL: number; PNEUMONIA: number };
  threshold: number;
}
export interface Evaluation {
  available: boolean;
  metrics: Metrics | null;
  plots: string[];
}
