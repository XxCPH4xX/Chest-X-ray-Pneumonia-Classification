"use client";
import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";
import { CheckCircle2, Info } from "lucide-react";
import { LabelBadge } from "./ui";
import { percentage } from "@/lib/api";
import type { Prediction } from "@/lib/types";

export function PredictionResult({
  result,
  showImages = true,
}: {
  result: Prediction;
  showImages?: boolean;
}) {
  const scores = [
    { label: "Normal", value: result.probabilities.NORMAL * 100 },
    { label: "Pneumonia", value: result.probabilities.PNEUMONIA * 100 },
  ];
  return (
    <section aria-label="Prediction result" className="result-section">
      <div className="section-top">
        <div>
          <span className="eyebrow">MODEL OUTPUT</span>
          <h2>Analysis result</h2>
        </div>
        {result.saved && (
          <span className="saved-label">
            <CheckCircle2 size={15} />
            Metadata saved
          </span>
        )}
      </div>
      <div className="result-grid">
        <div>
          <span className="subtle">Model classification</span>
          <div className="result-label">
            <LabelBadge label={result.predicted_class} />
          </div>
          <p className="caption">
            {result.predicted_class === "INCONCLUSIVE"
              ? "The score falls within the model’s configured validation-based uncertainty interval."
              : "This is a model classification, not a clinical finding."}
          </p>
          <dl className="details-list">
            <div>
              <dt>Decision threshold</dt>
              <dd>{percentage(result.threshold)}</dd>
            </div>
            <div>
              <dt>Architecture</dt>
              <dd>ResNet18</dd>
            </div>
            <div>
              <dt>Processing time</dt>
              <dd>{result.seconds.toFixed(2)} s</dd>
            </div>
            <div>
              <dt>Model version</dt>
              <dd className="mono">{result.model_version}</dd>
            </div>
          </dl>
        </div>
        <div>
          <h3>Model scores</h3>
          <div className="score-values">
            <span>
              Normal <strong>{percentage(result.probabilities.NORMAL)}</strong>
            </span>
            <span>
              Pneumonia{" "}
              <strong>{percentage(result.probabilities.PNEUMONIA)}</strong>
            </span>
          </div>
          <div
            className="score-chart"
            aria-label={`Normal score ${percentage(result.probabilities.NORMAL)}, Pneumonia score ${percentage(result.probabilities.PNEUMONIA)}`}
          >
            <ResponsiveContainer width="100%" height={132}>
              <BarChart
                data={scores}
                layout="vertical"
                margin={{ left: 0, right: 14, top: 10, bottom: 8 }}
              >
                <XAxis type="number" domain={[0, 100]} hide />
                <YAxis
                  type="category"
                  dataKey="label"
                  width={86}
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: "#586879", fontSize: 12 }}
                />
                <Bar
                  dataKey="value"
                  radius={[0, 4, 4, 0]}
                  barSize={22}
                  isAnimationActive={false}
                >
                  {scores.map((entry, index) => (
                    <Cell
                      key={entry.label}
                      fill={index ? "#8195a9" : "#168b7f"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="caption">
            Scores are not calibrated probabilities of disease. Image quality
            and differences from the training data can affect the result.
          </p>
        </div>
      </div>
      {showImages && result.images && (
        <>
          <div className="section-top explanation-heading">
            <div>
              <span className="eyebrow">GRAD-CAM</span>
              <h3>Model influence</h3>
            </div>
            <span className="subtle">Target: {result.gradcam_target}</span>
          </div>
          <div className="image-comparison">
            {[
              ["Model input", result.images.model_input],
              ["Influence heatmap", result.images.heatmap],
              ["Heatmap overlay", result.images.overlay],
            ].map(([title, src]) => (
              <figure key={title}>
                <div className="radiograph">
                  <img src={src} alt={title} />
                </div>
                <figcaption>{title}</figcaption>
              </figure>
            ))}
          </div>
          <div className="notice">
            <Info size={18} />
            <p>
              The map shows influence on the selected model score. It is not a
              verified lesion location or diagnostic explanation. All views use
              the model’s padded image coordinates.
            </p>
          </div>
        </>
      )}
    </section>
  );
}
