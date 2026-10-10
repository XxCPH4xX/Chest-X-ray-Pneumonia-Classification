import { hosted } from "@/lib/environment";
export const apiUrl = (path: string) =>
  `${process.env.NEXT_PUBLIC_API_URL || ""}${path}`;

export async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(apiUrl(path), { ...options, cache: "no-store" });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError")
      throw error;
    throw new Error(
      hosted
        ? "Cannot reach the analysis service. Please try again."
        : "Cannot reach the local API. Start the backend and try again.",
    );
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : `Request failed (${response.status}).`,
    );
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}
export const percentage = (value: number | null | undefined) =>
  value == null ? "Not available" : `${(value * 100).toFixed(1)}%`;
export const timestamp = (value: string) =>
  new Date(value).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
