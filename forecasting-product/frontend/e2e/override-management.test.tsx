import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { OverrideManagementPanel } from "@/components/sku/override-management-panel";
import { api, ApiError, setApiSession } from "@/lib/api-client";
import { ROLE_PERMISSIONS } from "@/lib/constants";
import { AuthProvider, useAuth } from "@/providers/auth-provider";
import type { OverrideItem, OverrideListResponse, Role, UpdateOverrideRequest } from "@/lib/types";

const realUpdate = api.updateOverride;

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((done, fail) => { resolve = done; reject = fail; });
  return { promise, resolve, reject };
}

function override(id = "OVR-A", oldSku = "old-one"): OverrideItem {
  return {
    override_id: id, old_sku: oldSku, new_sku: `new-${oldSku}`,
    proportion: 0.75, scenario: "custom", ramp_shape: "sigmoid",
    effective_date: "2026-01-02", notes: "stored notes", created_by: "original-planner",
    created_at: "2026-01-01T01:02:03", status: "approved",
    approved_by: "original-manager", approved_at: "2026-01-01T02:03:04",
  };
}

let items: OverrideItem[];
let nextRole: Role;
let list: ReturnType<typeof vi.spyOn<typeof api, "listOverrides">>;
let update: ReturnType<typeof vi.spyOn<typeof api, "updateOverride">>;
let create: ReturnType<typeof vi.spyOn<typeof api, "createOverride">>;
let remove: ReturnType<typeof vi.spyOn<typeof api, "deleteOverride">>;

function snapshot(): OverrideListResponse {
  return { count: items.length, overrides: items.map(row => ({ ...row })) };
}

function SessionControls() {
  const auth = useAuth();
  return <>
    <button disabled={auth.mode !== "production"} onClick={() => void auth.loginWithToken("opaque-first-session")}>
      Sign in fixture
    </button>
    <button disabled={auth.mode !== "production"} onClick={() => void auth.loginWithToken("opaque-replacement-session")}>
      Replace session
    </button>
    <button onClick={auth.logout}>Log out fixture</button>
    <OverrideManagementPanel />
  </>;
}

async function renderPanel(role: Role = "admin") {
  nextRole = role;
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  render(
    <QueryClientProvider client={client}>
      <AuthProvider><SessionControls /></AuthProvider>
    </QueryClientProvider>,
  );
  await waitFor(() => expect(screen.getByText("Sign in fixture")).toBeEnabled());
  fireEvent.click(screen.getByText("Sign in fixture"));
  await screen.findByText("2 override(s)");
  return client;
}

function row(id = "OVR-A") {
  const element = document.querySelector(`[data-override-id="${id}"]`);
  if (!(element instanceof HTMLElement)) throw new Error("Expected override row is missing");
  return within(element);
}

function edit(id = "OVR-A") {
  fireEvent.click(row(id).getByRole("button", { name: "Edit", exact: true }));
}

beforeEach(() => {
  items = [override(), override("OVR-B", "old-two")];
  nextRole = "admin";
  setApiSession(null);
  vi.spyOn(api, "authStatus").mockResolvedValue({ mode: "production", auth_required: true });
  vi.spyOn(api, "getCurrentUser").mockImplementation(async () => ({
    username: "verified-user", email: "", mode: "production", role: nextRole,
    permissions: ROLE_PERMISSIONS[nextRole], expires_at: Math.floor(Date.now() / 1000) + 3600,
  }));
  list = vi.spyOn(api, "listOverrides").mockImplementation(async () => snapshot());
  update = vi.spyOn(api, "updateOverride").mockImplementation(async (id, changes) => {
    items = items.map(row => row.override_id === id ? { ...row, ...changes } : row);
    return { override_id: id, status: "updated" };
  });
  create = vi.spyOn(api, "createOverride").mockImplementation(async body => {
    items.push({
      override_id: "OVR-C", old_sku: body.old_sku, new_sku: body.new_sku,
      proportion: body.proportion, scenario: body.scenario ?? "manual",
      ramp_shape: body.ramp_shape ?? "linear", notes: body.notes, effective_date: body.effective_date,
    });
    return { override_id: "OVR-C", status: "created" };
  });
  remove = vi.spyOn(api, "deleteOverride").mockImplementation(async id => {
    items = items.filter(row => row.override_id !== id);
    return { override_id: id, status: "deleted" };
  });
});

afterEach(() => {
  cleanup();
  setApiSession(null);
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("identity-preserving override editor", () => {
  it("uses the existing form, read-only SKUs and PUT to clear nullable fields and set zero", async () => {
    await renderPanel();
    const before = snapshot();
    edit();
    expect(screen.getByLabelText("Old SKU *")).toHaveAttribute("readonly");
    expect(screen.getByLabelText("New SKU *")).toHaveAttribute("readonly");
    expect(screen.getByLabelText("Old SKU *")).toHaveValue("old-one");
    expect(screen.getByLabelText("Ramp Shape")).toHaveValue("sigmoid");
    fireEvent.change(screen.getByLabelText("Proportion"), { target: { value: "0" } });
    fireEvent.change(screen.getByLabelText("Notes"), { target: { value: "" } });
    fireEvent.change(screen.getByLabelText("Effective Date"), { target: { value: "" } });
    fireEvent.click(screen.getByRole("button", { name: "Save", exact: true }));
    await waitFor(() => expect(screen.queryByText("Edit Override", { exact: true })).not.toBeInTheDocument());
    expect(update).toHaveBeenCalledExactlyOnceWith("OVR-A", {
      proportion: 0, notes: null, effective_date: null,
    });
    expect(create).not.toHaveBeenCalled();
    expect(remove).not.toHaveBeenCalled();
    expect(list).toHaveBeenCalledTimes(2);
    expect(screen.getByText("2 override(s)")).toBeVisible();
    expect(row().getByRole("cell", { name: "0", exact: true })).toBeVisible();
    expect(items).toEqual([
      { ...before.overrides[0], proportion: 0, notes: null, effective_date: null },
      before.overrides[1],
    ]);
  });

  it("omits unchanged fields so an independent server edit is not overwritten", async () => {
    await renderPanel();
    edit();
    items[0] = { ...items[0], scenario: "newer-server-scenario" };
    fireEvent.change(screen.getByLabelText("Notes"), { target: { value: "planner edit" } });
    fireEvent.click(screen.getByRole("button", { name: "Save", exact: true }));
    await waitFor(() => expect(row().getByText("newer-server-scenario")).toBeVisible());
    expect(update).toHaveBeenCalledExactlyOnceWith("OVR-A", { notes: "planner edit" });
  });

  it("creates with nullable blank fields as null without contaminating the edit DTO", async () => {
    await renderPanel();
    fireEvent.click(screen.getByRole("button", { name: "Add Override" }));
    expect(screen.getByLabelText("Old SKU *")).not.toHaveAttribute("readonly");
    fireEvent.change(screen.getByLabelText("Old SKU *"), { target: { value: "created-old" } });
    fireEvent.change(screen.getByLabelText("New SKU *"), { target: { value: "created-new" } });
    fireEvent.change(screen.getByLabelText("Proportion"), { target: { value: "0" } });
    fireEvent.click(screen.getByRole("button", { name: "Create Override" }));
    await screen.findByText("3 override(s)");
    expect(create).toHaveBeenCalledExactlyOnceWith({
      old_sku: "created-old", new_sku: "created-new", proportion: 0,
      scenario: "base", ramp_shape: "linear", notes: null, effective_date: null,
    });
    expect(update).not.toHaveBeenCalled();
  });

  it("rejects an empty patch and resets unsaved input on Cancel", async () => {
    await renderPanel();
    edit();
    fireEvent.click(screen.getByRole("button", { name: "Save", exact: true }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Change at least one editable field");
    expect(update).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText("Notes"), { target: { value: "not saved" } });
    fireEvent.click(screen.getByRole("button", { name: "Cancel", exact: true }));
    edit();
    expect(screen.getByLabelText("Notes")).toHaveValue("stored notes");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it.each(["", "-0.1", "1.1"])("does not turn invalid proportion %j into a valid zero", async value => {
    await renderPanel();
    edit();
    fireEvent.change(screen.getByLabelText("Proportion"), { target: { value } });
    fireEvent.click(screen.getByRole("button", { name: "Save", exact: true }));
    expect(await screen.findByRole("alert")).toHaveTextContent("finite number between 0 and 1");
    expect(update).not.toHaveBeenCalled();
  });

  it.each([400, 403, 404, 409, 500])("retains input and the original row after a failed Save (%s)", async status => {
    await renderPanel();
    const before = snapshot();
    update.mockRejectedValueOnce(new ApiError(status, `Rejected update ${status}`));
    edit();
    fireEvent.change(screen.getByLabelText("Notes"), { target: { value: "retry this input" } });
    fireEvent.click(screen.getByRole("button", { name: "Save", exact: true }));
    expect(await screen.findByRole("alert")).toHaveTextContent(`Rejected update ${status}`);
    expect(screen.getByLabelText("Notes")).toHaveValue("retry this input");
    expect(row().getByText("stored notes")).toBeVisible();
    expect(snapshot()).toEqual(before);
    expect(list).toHaveBeenCalledTimes(1);
    expect(create).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Save", exact: true }));
    await waitFor(() => expect(row().getByText("retry this input")).toBeVisible());
    expect(update).toHaveBeenCalledTimes(2);
    expect(screen.getByText("2 override(s)")).toBeVisible();
  });

  it("distinguishes an acknowledged Save with failed refresh from an uncommitted Save", async () => {
    await renderPanel();
    edit();
    list.mockRejectedValueOnce(new Error("read unavailable"));
    fireEvent.change(screen.getByLabelText("Notes"), { target: { value: "committed edit" } });
    fireEvent.click(screen.getByRole("button", { name: "Save", exact: true }));
    expect(await screen.findByRole("alert")).toHaveTextContent("was saved, but refreshing the list failed");
    expect(screen.getByLabelText("Notes")).toHaveValue("committed edit");
    expect(row().getByText("stored notes")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Refresh", exact: true }));
    await waitFor(() => expect(row().getByText("committed edit")).toBeVisible());
    expect(update).toHaveBeenCalledTimes(1);
  });

  it.each(["cancel", "other", "create"] as const)(
    "ignores old Save results after changing editor ownership (%s)", async replacement => {
      await renderPanel();
      const pending = deferred<{ override_id: string; status: string }>();
      update.mockReturnValueOnce(pending.promise);
      edit();
      fireEvent.change(screen.getByLabelText("Notes"), { target: { value: "late old edit" } });
      fireEvent.click(screen.getByRole("button", { name: "Save", exact: true }));
      expect(screen.getByLabelText("Notes")).toBeDisabled();
      if (replacement === "other") edit("OVR-B");
      else {
        fireEvent.click(screen.getByRole("button", { name: "Cancel", exact: true }));
        if (replacement === "create") fireEvent.click(screen.getByRole("button", { name: "Add Override" }));
      }
      await act(async () => pending.resolve({ override_id: "OVR-A", status: "updated" }));
      if (replacement === "cancel") expect(screen.queryByLabelText("Notes")).not.toBeInTheDocument();
      else expect(screen.getByLabelText("Old SKU *")).toHaveValue(replacement === "other" ? "old-two" : "");
      expect(row().getByText("stored notes")).toBeVisible();
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    },
  );

  it("ignores a stale failure and a stale post-Save read after switching rows", async () => {
    await renderPanel();
    const failed = deferred<{ override_id: string; status: string }>();
    update.mockReturnValueOnce(failed.promise);
    edit();
    fireEvent.change(screen.getByLabelText("Notes"), { target: { value: "first" } });
    fireEvent.click(screen.getByRole("button", { name: "Save", exact: true }));
    edit("OVR-B");
    await act(async () => failed.reject(new Error("stale failure")));
    expect(screen.queryByText("stale failure")).not.toBeInTheDocument();
    const staleRead = deferred<OverrideListResponse>();
    list.mockReturnValueOnce(staleRead.promise);
    fireEvent.change(screen.getByLabelText("Notes"), { target: { value: "second" } });
    fireEvent.click(screen.getByRole("button", { name: "Save", exact: true }));
    await waitFor(() => expect(list).toHaveBeenCalledTimes(2));
    edit();
    await act(async () => staleRead.resolve({ count: 99, overrides: [override("OVR-WRONG", "stale-row")] }));
    expect(screen.getByLabelText("Old SKU *")).toHaveValue("old-one");
    expect(screen.queryByText("99 override(s)")).not.toBeInTheDocument();
    expect(screen.queryByText("stale-row")).not.toBeInTheDocument();
  });

  it.each(["logout", "same-user-replacement"] as const)(
    "isolates pending mutations from a new authentication session (%s)", async transition => {
      await renderPanel();
      const pending = deferred<{ override_id: string; status: string }>();
      update.mockReturnValueOnce(pending.promise);
      edit();
      fireEvent.change(screen.getByLabelText("Notes"), { target: { value: "old private edit" } });
      fireEvent.click(screen.getByRole("button", { name: "Save", exact: true }));
      items = [override("OVR-NEW", "new-session-row")];
      if (transition === "logout") {
        fireEvent.click(screen.getByText("Log out fixture"));
        expect(screen.queryByText("Planner Override Management")).not.toBeInTheDocument();
        nextRole = "viewer";
        fireEvent.click(screen.getByText("Sign in fixture"));
      } else fireEvent.click(screen.getByText("Replace session"));
      await screen.findByText("1 override(s)");
      await act(async () => pending.resolve({ override_id: "OVR-A", status: "updated" }));
      expect(row("OVR-NEW").getByText("new-session-row")).toBeVisible();
      expect(screen.queryByText("old-one")).not.toBeInTheDocument();
      expect(screen.queryByDisplayValue("old private edit")).not.toBeInTheDocument();
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    },
  );

  it.each(["viewer", "manager", "planner", "admin"] as const)(
    "uses verified permissions rather than a local role picker (%s)", async role => {
      await renderPanel(role);
      const canEdit = role === "planner" || role === "admin";
      expect(screen.queryByRole("button", { name: "Add Override" }) !== null).toBe(canEdit);
      expect(row().queryByRole("button", { name: "Edit" }) !== null).toBe(canEdit);
      expect(row().queryByRole("button", { name: "Delete" }) !== null).toBe(role === "admin");
      expect(screen.queryByLabelText("Role")).not.toBeInTheDocument();
    },
  );

  it("deletes the selected ID and refreshes without a replacement create", async () => {
    await renderPanel();
    edit();
    fireEvent.click(row().getByRole("button", { name: "Delete" }));
    await screen.findByText("1 override(s)");
    expect(remove).toHaveBeenCalledExactlyOnceWith("OVR-A");
    expect(screen.queryByText("Edit Override", { exact: true })).not.toBeInTheDocument();
    expect(create).not.toHaveBeenCalled();
    expect(row("OVR-B").getByText("old-two")).toBeVisible();
  });
});

describe("override PUT client contract", () => {
  it("serializes zero/null but not omitted fields or immutable SKU fields", async () => {
    const changes: UpdateOverrideRequest = { proportion: 0, notes: null, effective_date: undefined };
    // @ts-expect-error SKU identity is not part of the update DTO.
    const forbidden: UpdateOverrideRequest = { old_sku: "replacement" };
    expect(forbidden).toHaveProperty("old_sku");
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify({ override_id: "OVR-A", status: "updated" })),
    );
    vi.stubGlobal("fetch", fetchMock);
    await realUpdate("OVR-A", changes);
    const [url, options] = fetchMock.mock.calls[0];
    expect(String(url)).toMatch(/\/overrides\/OVR-A$/);
    expect(options?.method).toBe("PUT");
    expect(JSON.parse(String(options?.body))).toEqual({ proportion: 0, notes: null });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it.each([409, 500, 502])("does not silently retry a failed PUT (%s)", async status => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(new Response("Rejected", { status }));
    vi.stubGlobal("fetch", fetchMock);
    await expect(realUpdate("OVR-A", { notes: "after" })).rejects.toMatchObject({ status });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("rejects a successful response that substitutes a different override identity", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify({ override_id: "OVR-REPLACEMENT", status: "updated" })),
    );
    vi.stubGlobal("fetch", fetchMock);
    await expect(realUpdate("OVR-A", { notes: "after" })).rejects.toThrow("different override ID");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
