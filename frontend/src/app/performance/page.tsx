"use client";
import { Download, Info } from "lucide-react";
import { EmptyState, ErrorNotice, Loading, PageHeading } from "@/components/ui";
import { apiUrl, percentage } from "@/lib/api";
import { useRemote } from "@/lib/use-remote";
import type { Evaluation, ModelInfo } from "@/lib/types";

const plotTitles: Record<string, string> = {
  confusion_matrix: "Confusion matrix",
  roc_curve: "ROC curve",
  precision_recall_curve: "Precision–recall curve",
  training_curves: "Training and validation curves",
};
export default function Performance() {
  const evaluation = useRemote<Evaluation>("/api/model/metrics");
  const model = useRemote<ModelInfo>("/api/model/info");
  const metrics = evaluation.data?.metrics;
  return (
    <>
      <PageHeading
        eyebrow="MEASURED RESULTS"
        title="Model performance."
        description="Held-out test results for the selected model."
      />
      {model.error && (
        <ErrorNotice message={model.error} retry={model.refresh} />
      )}
      <section className="card model-spec">
        <div>
          <span className="card-label">Architecture</span>
          <strong>ResNet18</strong>
        </div>
        <div>
          <span className="card-label">Model version</span>
          <strong className="mono">
            {model.data?.version || "Unavailable"}
          </strong>
        </div>
        <div>
          <span className="card-label">Dataset</span>
          <strong>Chest X-Ray Images (Pneumonia)</strong>
        </div>
      </section>
      {evaluation.loading ? (
        <Loading />
      ) : evaluation.error ? (
        <ErrorNotice message={evaluation.error} retry={evaluation.refresh} />
      ) : !metrics ? (
        <div className="card">
          <EmptyState title="No evaluation artifacts available">
            Complete model selection and held-out evaluation to see measured
            performance. Reports must match the loaded model.
          </EmptyState>
        </div>
      ) : (
        <>
          <div className="performance-metrics">
            {[
              ["Accuracy", metrics.accuracy],
              ["Sensitivity", metrics.sensitivity],
              ["Specificity", metrics.specificity],
              ["Precision", metrics.precision],
              ["F1 score", metrics.f1],
              ["ROC-AUC", metrics.roc_auc],
            ].map(([label, value]) => (
              <div className="card summary" key={String(label)}>
                <span className="card-label">{label}</span>
                <strong className="summary-value">
                  {percentage(value as number | null)}
                </strong>
              </div>
            ))}
          </div>
          <section className="card">
            <div className="section-top">
              <div>
                <span className="eyebrow">CLASS SUPPORT</span>
                <h2>Evaluation population</h2>
              </div>
              <span className="tag">
                Threshold {percentage(metrics.threshold)}
              </span>
            </div>
            <div className="support-row">
              <div>
                <strong>{metrics.support.NORMAL}</strong>
                <span>Normal images</span>
              </div>
              <div>
                <strong>{metrics.support.PNEUMONIA}</strong>
                <span>Pneumonia images</span>
              </div>
              <p>
                The original test split is held aside during model selection.
                Sensitivity is recall for PNEUMONIA; specificity is recall for
                NORMAL. Undefined metrics are displayed as unavailable.
              </p>
            </div>
            <div className="table-wrap">
              <table aria-label="Confusion matrix values">
                <thead>
                  <tr>
                    <th>Actual class</th>
                    <th>Predicted NORMAL</th>
                    <th>Predicted PNEUMONIA</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <th>NORMAL</th>
                    <td>{metrics.confusion_matrix[0][0]}</td>
                    <td>{metrics.confusion_matrix[0][1]}</td>
                  </tr>
                  <tr>
                    <th>PNEUMONIA</th>
                    <td>{metrics.confusion_matrix[1][0]}</td>
                    <td>{metrics.confusion_matrix[1][1]}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>
          <div className="plot-grid">
            {evaluation.data?.plots.map((name) => (
              <section className="card" key={name}>
                <div className="section-top">
                  <h3>{plotTitles[name] || name}</h3>
                  <a
                    className="icon-link"
                    href={apiUrl(`/api/model/plots/${name}`)}
                    download={`${name}.png`}
                    aria-label={`Download ${plotTitles[name]}`}
                  >
                    <Download size={17} />
                  </a>
                </div>
                <img
                  className="evaluation-plot"
                  src={apiUrl(`/api/model/plots/${name}`)}
                  alt={plotTitles[name]}
                />
              </section>
            ))}
          </div>
        </>
      )}
      <div className="notice">
        <Info size={20} />
        <div>
          <strong>Dataset performance has limits</strong>
          <p>
            The collection is pediatric and comes from a limited acquisition
            setting. Filename-derived patient groups cannot verify patient
            independence. Class imbalance, duplicate images, acquisition
            differences, and domain shift can affect generalization. These
            results do not establish clinical reliability.
          </p>
        </div>
      </div>
    </>
  );
}
