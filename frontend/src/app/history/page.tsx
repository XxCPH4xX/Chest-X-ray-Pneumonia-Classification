"use client";
import { hosted } from "@/lib/environment";
import { Suspense, useState } from "react";
import { useSearchParams } from "next/navigation";
import { ChevronLeft, ChevronRight, Search, Trash2, X } from "lucide-react";
import {
  EmptyState,
  ErrorNotice,
  LabelBadge,
  Loading,
  PageHeading,
} from "@/components/ui";
import { PredictionResult } from "@/components/prediction-result";
import { percentage, request, timestamp } from "@/lib/api";
import { useRemote } from "@/lib/use-remote";
import type { HistoryList, Prediction } from "@/lib/types";

function HistoryContent() {
  const searchParams = useSearchParams();
  const [search, setSearch] = useState("");
  const [label, setLabel] = useState("");
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<string | null>(
    searchParams.get("id"),
  );
  const [deleting, setDeleting] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const query = new URLSearchParams({
    search,
    offset: String(offset),
    limit: "10",
  });
  if (label) query.set("label", label);
  const history = useRemote<HistoryList>(`/api/predictions?${query}`);
  const detail = useRemote<Prediction>(
    selected ? `/api/predictions/${selected}` : null,
  );

  async function remove(identifier: string) {
    setDeleting(identifier);
    setError(null);
    try {
      await request(`/api/predictions/${identifier}`, { method: "DELETE" });
      if (selected === identifier) setSelected(null);
      if (history.data?.items.length === 1 && offset > 0)
        setOffset(offset - 10);
      else history.refresh();
    } catch (error) {
      setError(error instanceof Error ? error.message : "Deletion failed.");
    } finally {
      setDeleting(null);
    }
  }
  return (
    <>
      <PageHeading
        eyebrow={hosted ? "RECORDS" : "LOCAL RECORDS"}
        title="Prediction history."
        description="Review saved predictions. History stores result metadata, not images."
      />
      {error && <ErrorNotice message={error} />}
      <section className="card">
        <div className="history-controls">
          <div className="search-input">
            <Search size={18} />
            <input
              aria-label="Search prediction ID"
              placeholder="Search prediction ID…"
              value={search}
              onChange={(event) => {
                setSearch(event.target.value);
                setOffset(0);
              }}
            />
          </div>
          <select
            aria-label="Filter classification"
            value={label}
            onChange={(event) => {
              setLabel(event.target.value);
              setOffset(0);
            }}
          >
            <option value="">All classifications</option>
            <option value="NORMAL">Normal</option>
            <option value="PNEUMONIA">Pneumonia</option>
            <option value="INCONCLUSIVE">Inconclusive</option>
          </select>
        </div>
        {history.loading ? (
          <Loading />
        ) : history.error ? (
          <ErrorNotice message={history.error} retry={history.refresh} />
        ) : !history.data?.enabled ? (
          <EmptyState title="History is disabled">
            {hosted
              ? "Prediction history is not saved on this deployment."
              : "Enable local metadata history in the backend settings, then choose “Save result metadata” during an analysis. Uploaded images are never retained."}
          </EmptyState>
        ) : !history.data.items.length ? (
          <EmptyState
            title={
              search || label
                ? "No matching predictions"
                : "No saved predictions yet"
            }
          >
            {search || label
              ? "Try another ID or classification filter."
              : "Saved analysis results will appear here."}
          </EmptyState>
        ) : (
          <>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Prediction ID</th>
                    <th>Classification</th>
                    <th>Pneumonia score</th>
                    <th>Created</th>
                    <th>
                      <span className="sr-only">Actions</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {history.data.items.map((item) => (
                    <tr key={item.prediction_id}>
                      <td>
                        <button
                          className="text-button mono"
                          onClick={() => setSelected(item.prediction_id)}
                        >
                          {item.prediction_id.slice(0, 8)}
                        </button>
                      </td>
                      <td>
                        <LabelBadge label={item.predicted_class} />
                      </td>
                      <td>{percentage(item.probabilities.PNEUMONIA)}</td>
                      <td>{timestamp(item.created_at)}</td>
                      <td>
                        <button
                          className="icon-button"
                          aria-label={`Delete prediction ${item.prediction_id.slice(0, 8)}`}
                          disabled={deleting !== null}
                          onClick={() => remove(item.prediction_id)}
                        >
                          <Trash2 size={17} />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="pagination">
              <span>
                {offset + 1}–{offset + history.data.items.length} of{" "}
                {history.data.total}
              </span>
              <div>
                <button
                  className="icon-button"
                  aria-label="Previous page"
                  disabled={offset === 0}
                  onClick={() => setOffset(offset - 10)}
                >
                  <ChevronLeft size={18} />
                </button>
                <button
                  className="icon-button"
                  aria-label="Next page"
                  disabled={offset + 10 >= history.data.total}
                  onClick={() => setOffset(offset + 10)}
                >
                  <ChevronRight size={18} />
                </button>
              </div>
            </div>
          </>
        )}
      </section>
      {selected && (
        <section className="card history-detail">
          <button
            className="icon-button close-detail"
            aria-label="Close prediction details"
            onClick={() => setSelected(null)}
          >
            <X size={18} />
          </button>
          {detail.loading ? (
            <Loading />
          ) : detail.error ? (
            <ErrorNotice message={detail.error} />
          ) : (
            detail.data && (
              <>
                <PredictionResult result={detail.data} showImages={false} />
                <p className="caption">
                  Image previews and Grad-CAM are not retained in history.
                  Re-upload the image for a new explanation.
                </p>
              </>
            )
          )}
        </section>
      )}
    </>
  );
}
export default function HistoryPage() {
  return (
    <Suspense fallback={<Loading />}>
      <HistoryContent />
    </Suspense>
  );
}
