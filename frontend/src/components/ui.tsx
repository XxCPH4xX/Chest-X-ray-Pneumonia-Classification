import { AlertCircle, ArrowUpRight, LoaderCircle } from "lucide-react";
import Link from "next/link";

export function PageHeading({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow: string;
  title: string;
  description: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="page-heading">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {action}
    </div>
  );
}
export function ErrorNotice({
  message,
  retry,
}: {
  message: string;
  retry?: () => void;
}) {
  return (
    <div className="notice error" role="alert">
      <AlertCircle size={20} />
      <div>
        <strong>Something needs attention</strong>
        <p>{message}</p>
        {retry && (
          <button className="text-button" onClick={retry}>
            Try again
          </button>
        )}
      </div>
    </div>
  );
}
export function Loading({
  text = "Loading workspace data…",
}: {
  text?: string;
}) {
  return (
    <div className="loading" role="status">
      <LoaderCircle className="spin" size={20} />
      {text}
    </div>
  );
}
export function EmptyState({
  title,
  children,
  link,
}: {
  title: string;
  children: React.ReactNode;
  link?: { href: string; label: string };
}) {
  return (
    <div className="empty-state">
      <div className="empty-icon">
        <AlertCircle size={24} />
      </div>
      <h3>{title}</h3>
      <p>{children}</p>
      {link && (
        <Link className="button secondary" href={link.href}>
          {link.label}
          <ArrowUpRight size={16} />
        </Link>
      )}
    </div>
  );
}
export function LabelBadge({ label }: { label: string }) {
  return <span className={`label-badge ${label.toLowerCase()}`}>{label}</span>;
}
