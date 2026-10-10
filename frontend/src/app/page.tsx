"use client";
import { hosted } from "@/lib/environment";
import Link from "next/link";
import {
  ArrowRight,
  ArrowUpRight,
  Cpu,
  History,
  Layers,
  ScanLine,
} from "lucide-react";
import {
  EmptyState,
  ErrorNotice,
  LabelBadge,
  Loading,
  PageHeading,
} from "@/components/ui";
import { percentage, timestamp } from "@/lib/api";
import { useRemote } from "@/lib/use-remote";
import type { ModelInfo, HistoryList, Evaluation } from "@/lib/types";

export default function Dashboard() {
  const model = useRemote<ModelInfo>("/api/model/info");
  const history = useRemote<HistoryList>("/api/predictions?limit=4");
  const evaluation = useRemote<Evaluation>("/api/model/metrics");
  return (
    <>
      <PageHeading
        eyebrow={
          hosted ? "RESEARCH ENVIRONMENT" : "YOUR LOCAL RESEARCH ENVIRONMENT"
        }
        title="Research overview"
        description="Analyze chest X-rays and review model results."
        action={
          <Link href="/analysis" className="button primary">
            <ScanLine size={18} />
            Analyze an X-ray
          </Link>
        }
      />
      {model.error && (
        <ErrorNotice message={model.error} retry={model.refresh} />
      )}
      <div className="summary-grid">
        <div className="card summary">
          <div className="card-label">
            <Layers size={18} />
            Model status
          </div>
          <strong className="summary-value">
            {model.loading
              ? "Loading…"
              : model.data?.available
                ? "Ready for research"
                : "Unavailable"}
          </strong>
          <span className="subtle">
            {model.data?.available
              ? `${model.data.architecture} · selected checkpoint`
              : "An exported trained model is required"}
          </span>
        </div>
        <div className="card summary">
          <div className="card-label">
            <History size={18} />
            Saved predictions
          </div>
          <strong className="summary-value">
            {history.data
              ? history.data.enabled
                ? history.data.total
                : "History off"
              : "—"}
          </strong>
          <span className="subtle">
            {history.data?.enabled
              ? "Stored locally · metadata only"
              : "Saving is optional and disabled by default"}
          </span>
        </div>
        <div className="card summary">
          <div className="card-label">
            <Cpu size={18} />
            Execution environment
          </div>
          <strong className="summary-value">CPU only</strong>
          <span className="subtle">
            {hosted
              ? "Server inference · CPU only"
              : "Local inference · no cloud image processing"}
          </span>
        </div>
      </div>
      <div className="dashboard-grid">
        <section className="card workflow-card">
          <div className="section-top">
            <span className="eyebrow">IMAGE ANALYSIS</span>
            <ScanLine size={23} />
          </div>
          <h2>Analyze a chest X-ray</h2>
          <p>
            Upload a chest radiograph, inspect the classification scores, and
            explore a gradient-based influence map.
          </p>
          <div className="workflow-steps">
            {[
              ["01", "Upload", "Choose a JPEG or PNG"],
              ["02", "Analyze", "Run the ResNet18 model"],
              ["03", "Explore", "Compare scores and Grad-CAM"],
            ].map(([number, title, description]) => (
              <div className="workflow-step" key={number}>
                <span>{number}</span>
                <div>
                  <strong>{title}</strong>
                  <p>{description}</p>
                </div>
              </div>
            ))}
          </div>
          <Link href="/analysis" className="button primary">
            Start an analysis
            <ArrowRight size={18} />
          </Link>
        </section>
        <section className="card">
          <div className="section-top">
            <div>
              <span className="eyebrow">HELD-OUT TEST SET</span>
              <h2>Model evaluation</h2>
            </div>
            <Link
              href="/performance"
              aria-label="View model performance"
              className="icon-link"
            >
              <ArrowUpRight size={20} />
            </Link>
          </div>
          {evaluation.loading ? (
            <Loading />
          ) : evaluation.error ? (
            <ErrorNotice
              message={evaluation.error}
              retry={evaluation.refresh}
            />
          ) : evaluation.data?.metrics ? (
            <>
              <div className="metric-grid">
                {[
                  ["ROC-AUC", evaluation.data.metrics.roc_auc],
                  ["Sensitivity", evaluation.data.metrics.sensitivity],
                  ["Specificity", evaluation.data.metrics.specificity],
                  ["Accuracy", evaluation.data.metrics.accuracy],
                ].map(([label, value]) => (
                  <div key={String(label)}>
                    <span>{label}</span>
                    <strong>{percentage(value as number | null)}</strong>
                  </div>
                ))}
              </div>
              <p className="caption">
                Measured on the original held-out test set. Dataset results do
                not establish clinical reliability.
              </p>
              <Link href="/performance" className="text-link">
                Explore evaluation
                <ArrowRight size={16} />
              </Link>
            </>
          ) : (
            <EmptyState title="Evaluation is not available">
              Actual results appear after model selection and held-out
              evaluation.
            </EmptyState>
          )}
        </section>
      </div>
      <section className="card">
        <div className="section-top">
          <div>
            <span className="eyebrow">YOUR WORK</span>
            <h2>Recent predictions</h2>
          </div>
          <Link href="/history" className="text-link">
            View history
            <ArrowRight size={16} />
          </Link>
        </div>
        {history.loading ? (
          <Loading />
        ) : history.error ? (
          <ErrorNotice message={history.error} retry={history.refresh} />
        ) : history.data?.items.length ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Prediction</th>
                  <th>Model classification</th>
                  <th>Pneumonia score</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {history.data.items.map((item) => (
                  <tr key={item.prediction_id}>
                    <td>
                      <Link
                        href={`/history?id=${item.prediction_id}`}
                        className="text-link mono"
                      >
                        {item.prediction_id.slice(0, 8)}
                      </Link>
                    </td>
                    <td>
                      <LabelBadge label={item.predicted_class} />
                    </td>
                    <td>{percentage(item.probabilities.PNEUMONIA)}</td>
                    <td>{timestamp(item.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            title={
              history.data?.enabled ? "No saved predictions" : "History is off"
            }
            link={{ href: "/analysis", label: "Analyze an X-ray" }}
          >
            {history.data?.enabled
              ? "Save a prediction during analysis to find it here. Images are never stored."
              : hosted
                ? "This deployment does not save images or prediction history."
                : "Images and results are not saved by default. Local metadata history can be enabled in the backend settings."}
          </EmptyState>
        )}
      </section>
      <div className="methodology-strip">
        <div>
          <strong>Methods and limitations</strong>
          <p>
            Preprocessing, dataset separation, model selection, and the limits
            of explainability.
          </p>
        </div>
        <Link href="/about" className="text-link">
          Read methodology
          <ArrowUpRight size={18} />
        </Link>
      </div>
    </>
  );
}
