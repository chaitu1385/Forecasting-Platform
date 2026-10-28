"use client";

import { useState, useEffect, useCallback, useId } from "react";
import { useAsyncOperation } from "@/hooks/use-async-operation";
import { api } from "@/lib/api-client";
import { useAuth } from "@/providers/auth-provider";
import type {
  OverrideItem, OverrideListResponse, CreateOverrideRequest, UpdateOverrideRequest,
} from "@/lib/types";

interface OverrideForm {
  old_sku: string;
  new_sku: string;
  proportion: string;
  scenario: string;
  ramp_shape: string;
  effective_date: string;
  notes: string;
}

const EMPTY_OVERRIDE_FORM: OverrideForm = {
  old_sku: "",
  new_sku: "",
  proportion: "1",
  scenario: "base",
  ramp_shape: "linear",
  effective_date: "",
  notes: "",
};

export function OverrideManagementPanel() {
  const { user, hasPermission } = useAuth();
  const formId = useId();
  const [owner, setOwner] = useState(user);
  const [data, setData] = useState<OverrideListResponse | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [editing, setEditing] = useState<OverrideItem | null>(null);
  const [form, setForm] = useState<OverrideForm>({ ...EMPTY_OVERRIDE_FORM });
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const {
    run: load, reset: resetLoad, loading, error: loadError,
  } = useAsyncOperation<OverrideListResponse>();
  const {
    run: mutate, reset: resetMutation, loading: submitting, error: mutationError, setError,
  } = useAsyncOperation<OverrideListResponse>();

  const fetchOverrides = useCallback(async () => {
    const result = await load(() => api.listOverrides());
    if (result) setData(result);
  }, [load]);

  const resetEditor = useCallback(() => {
    resetMutation();
    setEditing(null);
    setForm({ ...EMPTY_OVERRIDE_FORM });
    setShowForm(false);
    setDeletingId(null);
  }, [resetMutation]);

  useEffect(() => {
    resetLoad();
    resetEditor();
    setData(null);
    setOwner(user);
    if (user) void fetchOverrides();
  }, [user, fetchOverrides, resetLoad, resetEditor]);

  const handleEdit = (row: OverrideItem) => {
    resetEditor();
    setEditing(row);
    setForm({
      old_sku: row.old_sku, new_sku: row.new_sku, proportion: String(row.proportion),
      scenario: row.scenario, ramp_shape: row.ramp_shape,
      effective_date: row.effective_date ?? "", notes: row.notes ?? "",
    });
    setShowForm(true);
  };

  const handleSave = async () => {
    if (!editing && (!form.old_sku || !form.new_sku)) {
      setError("Old SKU and New SKU are required.");
      return;
    }
    const proportion = Number(form.proportion);
    if (!form.proportion.trim() || !Number.isFinite(proportion) || proportion < 0 || proportion > 1) {
      setError("Proportion must be a finite number between 0 and 1.");
      return;
    }
    const effectiveDate = form.effective_date === "" ? null : form.effective_date;
    const notes = form.notes === "" ? null : form.notes;
    const changes: UpdateOverrideRequest = {};
    if (editing) {
      if (proportion !== editing.proportion) changes.proportion = proportion;
      if (form.scenario !== editing.scenario) changes.scenario = form.scenario;
      if (form.ramp_shape !== editing.ramp_shape) changes.ramp_shape = form.ramp_shape;
      if (form.effective_date !== (editing.effective_date ?? "")) changes.effective_date = effectiveDate;
      if (form.notes !== (editing.notes ?? "")) changes.notes = notes;
      if (Object.keys(changes).length === 0) {
        setError("Change at least one editable field before saving.");
        return;
      }
    }
    const create: CreateOverrideRequest = {
      ...form, proportion, effective_date: effectiveDate, notes,
    };
    resetLoad();
    setDeletingId(null);
    const result = await mutate(async () => {
      if (editing) await api.updateOverride(editing.override_id, changes);
      else await api.createOverride(create);
      try {
        return await api.listOverrides();
      } catch {
        throw new Error("The override was saved, but refreshing the list failed. Refresh before trying again.");
      }
    });
    if (!result) return;
    setData(result);
    resetEditor();
  };

  const handleDelete = async (id: string) => {
    resetLoad();
    setDeletingId(id);
    const result = await mutate(async () => {
      await api.deleteOverride(id);
      try {
        return await api.listOverrides();
      } catch {
        throw new Error("The override was deleted, but refreshing the list failed. Refresh before trying again.");
      }
    });
    if (!result) return;
    setData(result);
    if (editing?.override_id === id) resetEditor();
    setDeletingId(null);
  };

  const overrides = data?.overrides ?? [];
  const error = mutationError ?? loadError;
  // Hide the previous identity's rows even before its passive reset runs.
  if (!user || owner !== user) return null;

  return (
    <section className="space-y-4" aria-label="Planner Override Management">
      <h2 className="text-lg font-semibold">Planner Override Management</h2>
      <div className="rounded-lg border p-4 space-y-4">
        {/* Header row */}
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">
            {loading ? "Loading overrides..." : `${data?.count ?? 0} override(s)`}
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => { resetMutation(); void fetchOverrides(); }}
              disabled={loading || submitting}
              className="rounded-md border px-3 py-1.5 text-sm disabled:opacity-50"
            >
              Refresh
            </button>
            {hasPermission("create_override") && (
              <button
                onClick={() => {
                  resetEditor();
                  if (!showForm) setShowForm(true);
                }}
                className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90"
              >
                {showForm ? "Cancel" : "Add Override"}
              </button>
            )}
          </div>
        </div>

        {/* Error */}
        {error && (
          <div role="alert" className="rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive">
            {error}
          </div>
        )}

        {/* Create and edit share the same form. */}
        {showForm && (
          <div className="rounded-md border bg-muted/30 p-4 space-y-3">
            <h3 className="text-sm font-medium">{editing ? "Edit Override" : "New Override"}</h3>
            {editing && <p className="text-xs text-muted-foreground">{editing.override_id}</p>}
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
              <div className="space-y-1">
                <label htmlFor={`${formId}-old-sku`} className="text-xs font-medium">Old SKU *</label>
                <input
                  id={`${formId}-old-sku`}
                  type="text"
                  readOnly={editing !== null}
                  disabled={submitting}
                  value={form.old_sku}
                  onChange={(e) => setForm((prev) => ({ ...prev, old_sku: e.target.value }))}
                  className="w-full rounded-md border bg-background px-3 py-1.5 text-sm"
                  placeholder="e.g. SKU-001"
                />
              </div>
              <div className="space-y-1">
                <label htmlFor={`${formId}-new-sku`} className="text-xs font-medium">New SKU *</label>
                <input
                  id={`${formId}-new-sku`}
                  type="text"
                  readOnly={editing !== null}
                  disabled={submitting}
                  value={form.new_sku}
                  onChange={(e) => setForm((prev) => ({ ...prev, new_sku: e.target.value }))}
                  className="w-full rounded-md border bg-background px-3 py-1.5 text-sm"
                  placeholder="e.g. SKU-002"
                />
              </div>
              <div className="space-y-1">
                <label htmlFor={`${formId}-proportion`} className="text-xs font-medium">Proportion</label>
                <input
                  id={`${formId}-proportion`}
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  disabled={submitting}
                  value={form.proportion}
                  onChange={(e) => setForm((prev) => ({ ...prev, proportion: e.target.value }))}
                  className="w-full rounded-md border bg-background px-3 py-1.5 text-sm"
                />
              </div>
              <div className="space-y-1">
                <label htmlFor={`${formId}-scenario`} className="text-xs font-medium">Scenario</label>
                <input
                  id={`${formId}-scenario`}
                  type="text"
                  disabled={submitting}
                  value={form.scenario}
                  onChange={(e) => setForm((prev) => ({ ...prev, scenario: e.target.value }))}
                  className="w-full rounded-md border bg-background px-3 py-1.5 text-sm"
                  placeholder="base"
                />
              </div>
              <div className="space-y-1">
                <label htmlFor={`${formId}-ramp`} className="text-xs font-medium">Ramp Shape</label>
                <select
                  id={`${formId}-ramp`}
                  disabled={submitting}
                  value={form.ramp_shape}
                  onChange={(e) => setForm((prev) => ({ ...prev, ramp_shape: e.target.value }))}
                  className="w-full rounded-md border bg-background px-3 py-1.5 text-sm"
                >
                  <option value="linear">Linear</option>
                  <option value="step">Step</option>
                  <option value="exponential">Exponential</option>
                  {!["linear", "step", "exponential"].includes(form.ramp_shape) && (
                    <option value={form.ramp_shape}>{form.ramp_shape || "(empty)"}</option>
                  )}
                </select>
              </div>
              <div className="space-y-1">
                <label htmlFor={`${formId}-date`} className="text-xs font-medium">Effective Date</label>
                <input
                  id={`${formId}-date`}
                  type="date"
                  disabled={submitting}
                  value={form.effective_date}
                  onChange={(e) => setForm((prev) => ({ ...prev, effective_date: e.target.value }))}
                  className="w-full rounded-md border bg-background px-3 py-1.5 text-sm"
                />
              </div>
              <div className="space-y-1 sm:col-span-2 lg:col-span-3">
                <label htmlFor={`${formId}-notes`} className="text-xs font-medium">Notes</label>
                <input
                  id={`${formId}-notes`}
                  type="text"
                  disabled={submitting}
                  value={form.notes}
                  onChange={(e) => setForm((prev) => ({ ...prev, notes: e.target.value }))}
                  className="w-full rounded-md border bg-background px-3 py-1.5 text-sm"
                  placeholder="Optional notes"
                />
              </div>
            </div>
            <button
              onClick={handleSave}
              disabled={submitting}
              className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
            >
              {submitting && deletingId === null
                ? (editing ? "Saving..." : "Creating...")
                : (editing ? "Save" : "Create Override")}
            </button>
          </div>
        )}

        {/* Overrides table */}
        {loading && !data ? (
          <div className="space-y-2">
            <div className="h-8 animate-pulse rounded bg-muted" />
            <div className="h-8 animate-pulse rounded bg-muted" />
            <div className="h-8 animate-pulse rounded bg-muted" />
          </div>
        ) : overrides.length === 0 ? (
          <p className="text-sm text-muted-foreground">No overrides found.</p>
        ) : (
          <div className="overflow-x-auto rounded-md border">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-muted/50">
                  <th className="px-3 py-2 text-left font-medium">Old SKU</th>
                  <th className="px-3 py-2 text-left font-medium">New SKU</th>
                  <th className="px-3 py-2 text-left font-medium">Proportion</th>
                  <th className="px-3 py-2 text-left font-medium">Scenario</th>
                  <th className="px-3 py-2 text-left font-medium">Ramp Shape</th>
                  <th className="px-3 py-2 text-left font-medium">Effective Date</th>
                  <th className="px-3 py-2 text-left font-medium">Notes</th>
                  <th className="px-3 py-2 text-left font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {overrides.map((o) => (
                  <tr key={o.override_id} data-override-id={o.override_id} className="border-b last:border-0">
                    <td className="px-3 py-1.5">{o.old_sku}</td>
                    <td className="px-3 py-1.5">{o.new_sku}</td>
                    <td className="px-3 py-1.5">{o.proportion}</td>
                    <td className="px-3 py-1.5">{o.scenario}</td>
                    <td className="px-3 py-1.5">{o.ramp_shape}</td>
                    <td className="px-3 py-1.5">{o.effective_date ?? ""}</td>
                    <td className="px-3 py-1.5">{o.notes ?? ""}</td>
                    <td className="px-3 py-1.5">
                      {hasPermission("create_override") && (
                        <button
                          onClick={() => handleEdit(o)}
                          className="rounded px-2 py-1 text-xs font-medium hover:bg-muted"
                        >
                          Edit
                        </button>
                      )}
                      {hasPermission("delete_override") && (
                        <button
                          onClick={() => handleDelete(o.override_id)}
                          disabled={submitting}
                          className="rounded px-2 py-1 text-xs font-medium text-destructive hover:bg-destructive/10 disabled:opacity-50"
                        >
                          {submitting && deletingId === o.override_id ? "Deleting..." : "Delete"}
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
}
