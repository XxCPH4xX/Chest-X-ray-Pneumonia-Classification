import { hosted } from "@/lib/environment";
import { BookOpen, Cpu, Database, ScanLine, ShieldCheck } from "lucide-react";
import { PageHeading } from "@/components/ui";

export default function About() {
  return (
    <>
      <PageHeading
        eyebrow="METHODS"
        title="Methodology"
        description="Dataset preparation, training, evaluation, and limitations."
      />
      <div className="methodology-grid">
        <section className="card prose-card">
          <Database className="section-icon" size={24} />
          <h2>Dataset splits</h2>
          <p>
            The project uses the{" "}
            <a
              href="https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia"
              target="_blank"
              rel="noreferrer"
            >
              Kaggle Chest X-Ray Images (Pneumonia)
            </a>{" "}
            collection with NORMAL and PNEUMONIA labels.
          </p>
          <p>
            Images are checked for decoding errors, duplicates, and related
            filename groups. Validation uses a reproducible stratified group
            split of the original training set. The original test set is
            reserved for final evaluation.
          </p>
          <p>
            Filename groups are a conservative proxy. Authoritative patient
            metadata is not available, so patient independence cannot be
            verified.
          </p>
        </section>
        <section className="card prose-card">
          <ScanLine className="section-icon" size={24} />
          <h2>Preprocessing</h2>
          <p>
            Images become three-channel RGB and are resized to 224 × 224 with
            aspect-preserving black padding. Pixel values use ImageNet channel
            normalization.
          </p>
          <p>
            Training adds rotations up to 5° and translations up to 2%. There
            are no horizontal or vertical flips, aggressive crops, or heavy
            distortions. Validation, test evaluation, and uploaded-image
            inference share the same deterministic transformation.
          </p>
        </section>
        <section className="card prose-card">
          <Cpu className="section-icon" size={24} />
          <h2>Model development</h2>
          <p>
            ResNet18 starts with ImageNet weights. Its final layer produces one
            logit, trained with class-weighted binary cross-entropy. The initial
            run freezes the backbone and trains only the classification head.
          </p>
          <p>
            Validation ROC-AUC selects the checkpoint. A fixed decision
            threshold of 0.5 is recorded with the model. CPU-only training uses
            batches of eight, two compute threads, saved checkpoints, and early
            stopping. Notebooks call shared Python modules.
          </p>
        </section>
        <section className="card prose-card">
          <BookOpen className="section-icon" size={24} />
          <h2>Evaluation and explanations</h2>
          <p>
            The selected model is evaluated on the original held-out test set
            after selection is locked. Reports include class support, confusion
            matrix, ROC-AUC, sensitivity, specificity, precision, and F1.
          </p>
          <p>
            Grad-CAM uses real activations and gradients from the final
            convolutional block. Highlighted regions influence the selected
            score; they are not verified lesions or a causal explanation. A
            score is not a calibrated probability of disease.
          </p>
        </section>
      </div>
      <section className="card privacy-card">
        <ShieldCheck size={30} />
        <div>
          <span className="eyebrow">PRIVACY & APPROPRIATE USE</span>
          <h2>{hosted ? "Hosted processing" : "Local processing"}</h2>
          <p>
            {hosted
              ? "Images are sent to the hosted server for CPU analysis. The application does not save uploaded images or prediction history."
              : "Images are processed on this computer and are never stored by the application. Optional SQLite history stores scores and model metadata only. There are no third-party analytics on uploads."}
          </p>
          <p>
            This initial system is for educational research, not use with real
            patient information. It has not been clinically validated and must
            not be used to diagnose disease, replace radiologists, or make
            treatment decisions.
          </p>
        </div>
      </section>
    </>
  );
}
