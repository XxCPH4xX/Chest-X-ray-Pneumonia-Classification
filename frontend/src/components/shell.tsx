"use client";
import { hosted } from "@/lib/environment";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  BarChart3,
  BookOpen,
  Cpu,
  History,
  LayoutDashboard,
  ScanLine,
  ShieldCheck,
} from "lucide-react";

const navigation = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/analysis", label: "X-ray analysis", icon: ScanLine },
  { href: "/history", label: "Prediction history", icon: History },
  { href: "/performance", label: "Model performance", icon: BarChart3 },
  { href: "/about", label: "Methodology", icon: BookOpen },
];
export function Shell({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  return (
    <div className="workspace">
      <a href="#main-content" className="skip-link">
        Skip to content
      </a>
      <aside className="sidebar">
        <Link href="/" className="brand">
          <span className="brand-icon">
            <Activity size={24} />
          </span>
          <span>
            Chest X-ray Lab<small>RESEARCH WORKSPACE</small>
          </span>
        </Link>
        <div className="nav-label">WORKSPACE</div>
        <nav aria-label="Main navigation">
          {navigation.map(({ href, label, icon: Icon }) => (
            <Link
              key={href}
              href={href}
              aria-current={path === href ? "page" : undefined}
              className={path === href ? "nav-link active" : "nav-link"}
            >
              <Icon size={19} />
              {label}
            </Link>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="local-status">
            <span className="status-dot" />{" "}
            {hosted ? "Hosted environment" : "Local environment"}
          </div>
          <p>
            <Cpu size={15} /> CPU-only execution
          </p>
          <div className="privacy-note">
            <ShieldCheck size={19} />
            <span>
              {hosted
                ? "Images are sent to the server for analysis."
                : "Your images stay on this computer."}
            </span>
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <span>
            Workspace <span className="divider">/</span>{" "}
            <strong>
              {navigation.find((item) => item.href === path)?.label ||
                "Research"}
            </strong>
          </span>
          <span className="research-tag">
            <span className="status-dot" /> RESEARCH USE ONLY
          </span>
        </header>
        <main id="main-content">{children}</main>
        <footer>
          For research and education. Not a diagnostic tool or a substitute for
          professional clinical assessment.
        </footer>
      </div>
    </div>
  );
}
