import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test, vi } from "vitest";
import Analysis from "@/app/analysis/page";

vi.mock("@/lib/environment", () => ({ hosted: true, uploadLimitMiB: 3 }));

test("hosted analysis explains server processing and rejects oversized uploads", async () => {
  const fetchMock = vi
    .fn()
    .mockResolvedValue(
      new Response(JSON.stringify({ available: true, history_enabled: false })),
    );
  vi.stubGlobal("fetch", fetchMock);
  render(<Analysis />);
  expect(
    screen.getByText(/Sent to the server for analysis/),
  ).toBeInTheDocument();
  expect(screen.getByText(/Up to 3 MiB/)).toBeInTheDocument();
  const file = new File([new Uint8Array(3 * 1024 * 1024 + 1)], "large.png", {
    type: "image/png",
  });
  await userEvent.upload(screen.getByLabelText("Choose X-ray image"), file);
  expect(screen.getByRole("alert")).toHaveTextContent("3 MiB upload limit");
  expect(screen.getByRole("button", { name: "Analyze image" })).toBeDisabled();
  expect(fetchMock.mock.calls.some(([url]) => url.endsWith("/predict"))).toBe(
    false,
  );
});
