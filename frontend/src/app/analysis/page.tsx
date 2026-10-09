"use client";
import { useEffect, useRef, useState } from "react";
import {
  FileImage,
  LoaderCircle,
  LockKeyhole,
  ScanLine,
  UploadCloud,
  X,
} from "lucide-react";
import { ErrorNotice, PageHeading } from "@/components/ui";
import { PredictionResult } from "@/components/prediction-result";
import { request } from "@/lib/api";
import { useRemote } from "@/lib/use-remote";
import type { ModelInfo, Prediction } from "@/lib/types";

export default function Analysis() {
  const model = useRemote<ModelInfo>("/api/model/info");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [explain, setExplain] = useState(true);
  const [save, setSave] = useState(false);
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const activeRequest = useRef<AbortController | null>(null);
  useEffect(
    () => () => {
      activeRequest.current?.abort();
    },
    [],
  );
  useEffect(
    () => () => {
      if (preview) URL.revokeObjectURL(preview);
    },
    [preview],
  );

  function selectFile(next: File | null) {
    activeRequest.current?.abort();
    setBusy(false);
    setResult(null);
    setError(null);
    setFile(null);
    setPreview(null);
    if (input.current) input.current.value = "";
    if (!next) return;
    if (!["image/png", "image/jpeg"].includes(next.type)) {
      setError("Choose a JPEG or PNG image.");
      return;
    }
    if (next.size > 8 * 1024 * 1024) {
      setError("Image exceeds the 8 MiB upload limit.");
      return;
    }
    setFile(next);
    setPreview(URL.createObjectURL(next));
  }
  async function analyze() {
    if (!file) return;
    activeRequest.current?.abort();
    const controller = new AbortController();
    activeRequest.current = controller;
    setBusy(true);
    setError(null);
    setResult(null);
    const body = new FormData();
    body.append("file", file);
    body.append("explain", String(explain));
    body.append("save_history", String(save && model.data?.history_enabled));
    try {
      const prediction = await request<Prediction>("/api/predict", {
        method: "POST",
        body,
        signal: controller.signal,
      });
      if (!controller.signal.aborted) setResult(prediction);
    } catch (error) {
      if (!controller.signal.aborted)
        setError(error instanceof Error ? error.message : "Analysis failed.");
    } finally {
      if (!controller.signal.aborted) setBusy(false);
    }
  }
  return (
    <>
      <PageHeading
        eyebrow="IMAGE ANALYSIS"
        title="X-ray analysis"
        description="Upload an X-ray to view model scores and Grad-CAM."
      />
      {model.error && (
        <ErrorNotice message={model.error} retry={model.refresh} />
      )}
      {model.data && !model.data.available && (
        <div className="notice" role="status">
          <ScanLine size={21} />
          <div>
            <strong>Model unavailable</strong>
            <p>
              Complete training, held-out evaluation, and model export before
              analyzing images. No prediction will be generated without a
              trained model.
            </p>
          </div>
        </div>
      )}
      <div className="analysis-grid">
        <section className="card upload-card">
          <div className="section-top">
            <div>
              <span className="eyebrow">01 · INPUT</span>
              <h2>Upload a radiograph</h2>
            </div>
            <span className="tag">JPEG / PNG</span>
          </div>
          <input
            ref={input}
            type="file"
            accept="image/jpeg,image/png"
            aria-label="Choose X-ray image"
            className="sr-only"
            onChange={(event) => selectFile(event.target.files?.[0] || null)}
          />
          <div
            className={`dropzone ${dragging ? "dragging" : ""} ${preview ? "has-image" : ""}`}
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragging(false);
              selectFile(event.dataTransfer.files[0] || null);
            }}
          >
            {preview ? (
              <>
                <img src={preview} alt="Selected chest X-ray preview" />
                <button
                  className="remove-image"
                  aria-label="Remove image"
                  onClick={() => selectFile(null)}
                >
                  <X size={18} />
                </button>
              </>
            ) : (
              <>
                <span className="upload-icon">
                  <UploadCloud size={30} />
                </span>
                <h3>Drop an X-ray image here</h3>
                <p>or select a file from your computer</p>
                <button
                  className="button secondary"
                  onClick={() => input.current?.click()}
                >
                  Browse files
                </button>
                <small>Up to 8 MiB · 4096 px per side · 12 MP</small>
              </>
            )}
          </div>
          {file && (
            <div className="file-row">
              <FileImage size={20} />
              <div>
                <strong>{file.name}</strong>
                <small>{(file.size / 1024).toFixed(0)} KiB</small>
              </div>
              <button
                className="text-button"
                onClick={() => input.current?.click()}
              >
                Replace
              </button>
            </div>
          )}
          <div className="privacy-inline">
            <LockKeyhole size={15} />
            <span>Processed locally. The original image is never saved.</span>
          </div>
        </section>
        <section className="card options-card">
          <span className="eyebrow">02 · OPTIONS</span>
          <h2>Analysis settings</h2>
          <label className="option-row">
            <div>
              <strong>Include Grad-CAM</strong>
              <p>Generate a heatmap of model influence.</p>
            </div>
            <input
              type="checkbox"
              checked={explain}
              onChange={(event) => setExplain(event.target.checked)}
              disabled={busy}
            />
          </label>
          <label className="option-row">
            <div>
              <strong>Save result metadata</strong>
              <p>
                {model.data?.history_enabled
                  ? "Keep scores in local history. Images are not stored."
                  : "History is disabled in the backend settings."}
              </p>
            </div>
            <input
              type="checkbox"
              checked={save}
              onChange={(event) => setSave(event.target.checked)}
              disabled={busy || !model.data?.history_enabled}
            />
          </label>
          <dl className="details-list">
            <div>
              <dt>Architecture</dt>
              <dd>ResNet18</dd>
            </div>
            <div>
              <dt>Execution</dt>
              <dd>CPU · local</dd>
            </div>
            <div>
              <dt>Input resolution</dt>
              <dd>
                {model.data?.preprocessing
                  ? `${model.data.preprocessing.image_size} × ${model.data.preprocessing.image_size}`
                  : "Unavailable"}
              </dd>
            </div>
          </dl>
          <button
            className="button primary analyze-button"
            disabled={!file || busy || !model.data?.available}
            onClick={analyze}
          >
            {busy ? (
              <>
                <LoaderCircle size={18} className="spin" />
                Analyzing on CPU…
              </>
            ) : (
              <>
                <ScanLine size={18} />
                Analyze image
              </>
            )}
          </button>
          <p className="caption">
            For chest radiograph research only. The model does not verify that
            an upload is an X-ray or suitable for analysis.
          </p>
        </section>
      </div>
      {busy && (
        <div className="notice" role="status">
          <LoaderCircle className="spin" size={20} />
          <p>Running the model locally. CPU analysis can take a few seconds.</p>
        </div>
      )}
      {error && <ErrorNotice message={error} />}
      {result && (
        <div className="card">
          <PredictionResult result={result} showImages={explain} />
        </div>
      )}
    </>
  );
}
