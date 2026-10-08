import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, test, vi } from "vitest";
import Analysis from "@/app/analysis/page";
import type { Prediction } from "@/lib/types";

const info = {
  available: true,
  history_enabled: false,
  preprocessing: { image_size: 224 },
};
const prediction: Prediction = {
  prediction_id: "test-id",
  created_at: "2026-10-09T00:00:00Z",
  predicted_class: "NORMAL",
  probabilities: { NORMAL: 0.8, PNEUMONIA: 0.2 },
  threshold: 0.5,
  architecture: "resnet18",
  model_version: "test-model",
  gradcam_available: false,
  gradcam_target: "NORMAL",
  uncertainty_policy: null,
  seconds: 0.1,
  saved: false,
  images: null,
};
const file = new File(["test bytes"], "xray.png", { type: "image/png" });
const json = (value: unknown, status = 200) =>
  new Response(JSON.stringify(value), {
    status,
    headers: { "Content-Type": "application/json" },
  });
let fetchMock: ReturnType<typeof vi.fn>;
beforeEach(() => {
  fetchMock = vi.fn().mockResolvedValue(json(info));
  vi.stubGlobal("fetch", fetchMock);
});

test("select, preview, replace and remove an image", async () => {
  render(<Analysis />);
  await userEvent.upload(screen.getByLabelText("Choose X-ray image"), file);
  expect(
    screen.getByAltText("Selected chest X-ray preview"),
  ).toBeInTheDocument();
  expect(screen.getByText("xray.png")).toBeInTheDocument();
  const replacement = new File(["bytes"], "replacement.png", {
    type: "image/png",
  });
  await userEvent.upload(
    screen.getByLabelText("Choose X-ray image"),
    replacement,
  );
  expect(screen.getByText("replacement.png")).toBeInTheDocument();
  expect(screen.queryByText("xray.png")).not.toBeInTheDocument();
  await userEvent.click(screen.getByLabelText("Remove image"));
  expect(
    screen.queryByAltText("Selected chest X-ray preview"),
  ).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Analyze image" })).toBeDisabled();
});

test("drag and drop validates unsupported content types and file size", async () => {
  render(<Analysis />);
  fireEvent.drop(screen.getByText("Drop an X-ray image here").parentElement!, {
    dataTransfer: {
      files: [new File(["bad"], "x.gif", { type: "image/gif" })],
    },
  });
  expect(screen.getByRole("alert")).toHaveTextContent("Choose a JPEG or PNG");
  const huge = new File([new Uint8Array(8 * 1024 * 1024 + 1)], "large.png", {
    type: "image/png",
  });
  await userEvent.upload(screen.getByLabelText("Choose X-ray image"), huge);
  expect(screen.getByRole("alert")).toHaveTextContent("8 MiB");
});

test("shows loading and renders only the returned prediction", async () => {
  let resolve!: (response: Response) => void;
  fetchMock.mockImplementation((url: string) =>
    url.endsWith("/predict")
      ? new Promise<Response>((done) => {
          resolve = done;
        })
      : Promise.resolve(json(info)),
  );
  render(<Analysis />);
  await userEvent.upload(screen.getByLabelText("Choose X-ray image"), file);
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Analyze image" })).toBeEnabled(),
  );
  await userEvent.click(screen.getByRole("button", { name: "Analyze image" }));
  expect(
    screen.getByRole("button", { name: /Analyzing on CPU/ }),
  ).toBeDisabled();
  expect(screen.queryByText("NORMAL")).not.toBeInTheDocument();
  await act(async () => resolve(json(prediction)));
  expect(await screen.findByText("NORMAL")).toBeInTheDocument();
  expect(screen.getByText("80.0%")).toBeInTheDocument();
  expect(screen.getByText(/not calibrated probabilities/)).toBeInTheDocument();
  const body = fetchMock.mock.calls.find((call) =>
    call[0].endsWith("/predict"),
  )![1].body as FormData;
  expect(body.get("save_history")).toBe("false");
});

test("server errors and model unavailable are explicit", async () => {
  fetchMock.mockResolvedValue(json({ ...info, available: false }));
  render(<Analysis />);
  expect(await screen.findByText("Model unavailable")).toBeInTheDocument();
  await userEvent.upload(screen.getByLabelText("Choose X-ray image"), file);
  expect(screen.getByRole("button", { name: "Analyze image" })).toBeDisabled();
});

test("failed inference exposes the API error without a result", async () => {
  fetchMock.mockImplementation((url: string) =>
    Promise.resolve(
      url.endsWith("/predict")
        ? json({ detail: "Image is corrupt." }, 400)
        : json(info),
    ),
  );
  render(<Analysis />);
  await userEvent.upload(screen.getByLabelText("Choose X-ray image"), file);
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Analyze image" })).toBeEnabled(),
  );
  await userEvent.click(screen.getByRole("button", { name: "Analyze image" }));
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Image is corrupt.",
  );
  expect(
    screen.queryByRole("region", { name: "Prediction result" }),
  ).not.toBeInTheDocument();
});

test("replacing an image cancels an in-flight result", async () => {
  let resolve!: (response: Response) => void;
  fetchMock.mockImplementation((url: string) =>
    url.endsWith("/predict")
      ? new Promise<Response>((done) => {
          resolve = done;
        })
      : Promise.resolve(json(info)),
  );
  render(<Analysis />);
  await userEvent.upload(screen.getByLabelText("Choose X-ray image"), file);
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Analyze image" })).toBeEnabled(),
  );
  await userEvent.click(screen.getByRole("button", { name: "Analyze image" }));
  await userEvent.click(screen.getByLabelText("Remove image"));
  await act(async () => resolve(json(prediction)));
  expect(screen.queryByText("NORMAL")).not.toBeInTheDocument();
});
