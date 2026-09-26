"""Typed declarative aggregation-plan schema and structural validation.

Plans describe generic relational operations over artifact tables without
task-specific entity names, expected values, or template rules.

Plan shape (validated structurally here, independently of table contents):

  {
    "sources": [{"artifact_id": "art003", "as": "tasks"}, ...],   # bind artifacts -> tables
    "steps":   [ <op>, ... ],                                     # each writes a new named table
    "output":  {"from": "<table>", "kind": "scalar"|"table", "field": "<f>"?}
  }

Generic ops (each reads a named table, writes `into` a new name):
  filter    {op, table, where:[<pred>...], into}          # AND of typed predicates
  cast      {op, table, casts:[{field, to}...], into}     # to: datetime|int|float|string
  project   {op, table, fields:[...], into}               # select/rename-free projection
  group_by  {op, table, keys:[...], aggs:[<agg>...], into} # keys may be [] (whole-table)
  argmin/argmax {op, table, field, tie_break:[<sort>]?, into}   # single-row result
  sort      {op, table, by:[<sort>...], into}             # stable
  top_k     {op, table, by:[<sort>...], k:int, into}      # stable head
  join      {op, left, right, on:[{left, right}...], how:"inner", into}   # equi-join
  union_all {op, tables:[t1, t2, ...], into}              # concat schema/type-identical
                                                          # tables, stable declared order

  <pred> = {field, cmp: eq|ne|lt|le|gt|ge|in|contains|exists, value?}
  <agg>  = {fn: count|sum|mean|min|max, field?, into}     # count needs no field
  <sort> = {field, dir: asc|desc}

"""
from __future__ import annotations

from typing import Any

MAX_SOURCES = 8
MAX_STEPS = 24
MAX_FIELDS = 30
MAX_PREDS = 12

CMPS = {"eq", "ne", "lt", "le", "gt", "ge", "in", "contains", "exists"}
AGG_FNS = {"count", "sum", "mean", "min", "max"}
DIRS = {"asc", "desc"}
CAST_TYPES = {"datetime", "int", "float", "string"}

# --------------------------------------------------------------------------- #
# Canonical DSL grammar for the aggregation PLANNER prompt. Kept HERE, beside the
# constants + validator, so the prompt and `validate_agg_plan` cannot silently drift.
# The domain-neutral example uses inventory rows -> group/count -> argmin.
# --------------------------------------------------------------------------- #
GRAMMAR = (
    "AGGREGATION PLAN — canonical JSON:\n"
    '{"sources":[{"artifact_id":"<id>","as":"<table>"}], "steps":[<op>...], '
    '"output":{"from":"<table>","kind":"scalar"|"table","field":"<f>"}}  '
    "(output.field is REQUIRED when kind=scalar; the source table must then have exactly one row)\n"
    "Each op reads named table(s) and writes a NEW table named by `into`:\n"
    '  filter:    {"op":"filter","table":T,"where":[{"field":F,"cmp":C,"value":V}...],"into":T2}\n'
    f"             cmp C in {sorted(CMPS)}; `exists` takes no value; lt/le/gt/ge need a numeric\n"
    "             or (after cast) datetime column — never a raw string column\n"
    '  cast:      {"op":"cast","table":T,"casts":[{"field":F,"to":TY}...],"into":T2}   '
    f"TY in {sorted(CAST_TYPES)}; datetime is strict ISO-8601\n"
    '  project:   {"op":"project","table":T,"fields":[F...],"into":T2}\n'
    '  group_by:  {"op":"group_by","table":T,"keys":[F...](may be []),'
    '"aggs":[{"fn":FN,"field":F?,"into":N}...],"into":T2}   '
    f"fn in {sorted(AGG_FNS)}; count needs no field\n"
    '  argmin/argmax: {"op":"argmin"|"argmax","table":T,"field":F,'
    '"tie_break":[{"field":F,"dir":D}...]?,"into":T2}   single-row result; dir in ["asc","desc"]\n'
    '  sort:      {"op":"sort","table":T,"by":[{"field":F,"dir":D}...],"into":T2}\n'
    '  top_k:     {"op":"top_k","table":T,"by":[{"field":F,"dir":D}...],"k":<int>,"into":T2}\n'
    '  join:      {"op":"join","left":T,"right":T2,"on":[{"left":F,"right":F}...],'
    '"how":"inner","into":T3}\n'
    '  union_all: {"op":"union_all","tables":[T...],"into":T2}   '
    "concatenate schema-identical tables in the listed order (mixed int/float columns widen "
    "to float; other type mixes are rejected)\n"
    "Rules: NEVER include a source whose catalog n_rows is 0 in union_all/join (an empty "
    "artifact has no columns and the plan is rejected). Use ONLY column names that appear in "
    "the catalog `columns` for that artifact — inventing a column rejects the plan."
)

EXAMPLE_PLAN = {
    "sources": [{"artifact_id": "art001", "as": "inventory"}],
    "steps": [
        {"op": "filter", "table": "inventory",
         "where": [{"field": "in_stock", "cmp": "eq", "value": True}], "into": "avail"},
        {"op": "group_by", "table": "avail", "keys": ["warehouse"],
         "aggs": [{"fn": "count", "into": "n"}], "into": "by_wh"},
        {"op": "argmin", "table": "by_wh", "field": "n",
         "tie_break": [{"field": "warehouse", "dir": "asc"}], "into": "least"},
    ],
    "output": {"from": "least", "kind": "scalar", "field": "warehouse"},
}


class AggPlanError(ValueError):
    """A malformed aggregation plan (fail loud)."""


def _name(v: Any, what: str) -> str:
    if not isinstance(v, str) or not v:
        raise AggPlanError(f"{what} must be a non-empty string, got {v!r}")
    return v


def _fields(v: Any, what: str) -> list[str]:
    if not isinstance(v, list) or not v or len(v) > MAX_FIELDS or not all(isinstance(f, str) and f for f in v):
        raise AggPlanError(f"{what} must be a non-empty list of <= {MAX_FIELDS} field names")
    return list(v)


def _sort_list(v: Any, what: str) -> list[dict]:
    if not isinstance(v, list) or not v or len(v) > MAX_FIELDS:
        raise AggPlanError(f"{what} must be a non-empty list of sort keys")
    out = []
    for s in v:
        if not isinstance(s, dict) or "field" not in s:
            raise AggPlanError(f"{what}: each sort key needs a field: {s!r}")
        d = s.get("dir", "asc")
        if d not in DIRS:
            raise AggPlanError(f"{what}: dir must be asc/desc, got {d!r}")
        out.append({"field": _name(s["field"], "sort field"), "dir": d})
    return out


def _pred(p: Any) -> dict:
    if not isinstance(p, dict) or "field" not in p or "cmp" not in p:
        raise AggPlanError(f"predicate needs field+cmp: {p!r}")
    cmp = p["cmp"]
    if cmp not in CMPS:
        raise AggPlanError(f"unknown cmp {cmp!r} (allowed: {sorted(CMPS)})")
    out = {"field": _name(p["field"], "predicate field"), "cmp": cmp}
    if cmp != "exists":
        if "value" not in p:
            raise AggPlanError(f"predicate cmp {cmp!r} needs a value: {p!r}")
        out["value"] = p["value"]
    return out


def _agg(a: Any) -> dict:
    if not isinstance(a, dict) or "fn" not in a or "into" not in a:
        raise AggPlanError(f"agg needs fn+into: {a!r}")
    fn = a["fn"]
    if fn not in AGG_FNS:
        raise AggPlanError(f"unknown agg fn {fn!r} (allowed: {sorted(AGG_FNS)})")
    out = {"fn": fn, "into": _name(a["into"], "agg into")}
    if fn == "count":
        if a.get("field") is not None:
            out["field"] = _name(a["field"], "count field")   # count of non-null field (optional)
    else:
        out["field"] = _name(a.get("field"), f"{fn} field")   # sum/mean/min/max require a field
    return out


def _validate_step(s: Any, defined: set[str]) -> dict:
    if not isinstance(s, dict) or "op" not in s or "into" not in s:
        raise AggPlanError(f"step needs op+into: {s!r}")
    op, into = s["op"], _name(s["into"], "into")

    def src(key="table"):
        t = _name(s.get(key), f"{op}.{key}")
        if t not in defined:
            raise AggPlanError(f"{op}: table {t!r} not defined before this step")
        return t

    if op == "filter":
        where = s.get("where", [])
        if not isinstance(where, list) or not where or len(where) > MAX_PREDS:
            raise AggPlanError("filter.where must be a non-empty list of predicates")
        out = {"op": op, "table": src(), "where": [_pred(p) for p in where], "into": into}
    elif op == "cast":
        casts = s.get("casts", [])
        if not isinstance(casts, list) or not casts or len(casts) > MAX_FIELDS:
            raise AggPlanError("cast.casts must be a non-empty list")
        cc = []
        for c in casts:
            if not isinstance(c, dict) or "field" not in c or "to" not in c:
                raise AggPlanError(f"cast entry needs field+to: {c!r}")
            if c["to"] not in CAST_TYPES:
                raise AggPlanError(f"cast.to must be one of {sorted(CAST_TYPES)}, got {c['to']!r}")
            cc.append({"field": _name(c["field"], "cast.field"), "to": c["to"]})
        out = {"op": op, "table": src(), "casts": cc, "into": into}
    elif op == "union_all":
        tabs = s.get("tables", [])
        if not isinstance(tabs, list) or len(tabs) < 2 or len(tabs) > MAX_SOURCES:
            raise AggPlanError("union_all.tables must list >= 2 defined tables")
        names = []
        for t in tabs:
            nm = _name(t, "union_all.table")
            if nm not in defined:
                raise AggPlanError(f"union_all: table {nm!r} not defined before this step")
            names.append(nm)
        out = {"op": op, "tables": names, "into": into}
    elif op == "project":
        out = {"op": op, "table": src(), "fields": _fields(s.get("fields"), "project.fields"), "into": into}
    elif op == "group_by":
        keys = s.get("keys", [])
        if not isinstance(keys, list) or len(keys) > MAX_FIELDS or not all(isinstance(k, str) and k for k in keys):
            raise AggPlanError("group_by.keys must be a list of field names (may be empty)")
        aggs = s.get("aggs", [])
        if not isinstance(aggs, list) or not aggs or len(aggs) > MAX_FIELDS:
            raise AggPlanError("group_by.aggs must be a non-empty list")
        out = {"op": op, "table": src(), "keys": list(keys), "aggs": [_agg(a) for a in aggs], "into": into}
    elif op in ("argmin", "argmax"):
        out = {"op": op, "table": src(), "field": _name(s.get("field"), f"{op}.field"), "into": into}
        if s.get("tie_break") is not None:
            out["tie_break"] = _sort_list(s["tie_break"], f"{op}.tie_break")
    elif op == "sort":
        out = {"op": op, "table": src(), "by": _sort_list(s.get("by"), "sort.by"), "into": into}
    elif op == "top_k":
        k = s.get("k")
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise AggPlanError(f"top_k.k must be a positive int, got {k!r}")
        out = {"op": op, "table": src(), "by": _sort_list(s.get("by"), "top_k.by"), "k": k, "into": into}
    elif op == "join":
        on = s.get("on", [])
        if not isinstance(on, list) or not on or len(on) > MAX_FIELDS:
            raise AggPlanError("join.on must be a non-empty list of {left,right}")
        pairs = []
        for pr in on:
            if not isinstance(pr, dict) or "left" not in pr or "right" not in pr:
                raise AggPlanError(f"join.on pair needs left+right: {pr!r}")
            pairs.append({"left": _name(pr["left"], "join left"), "right": _name(pr["right"], "join right")})
        how = s.get("how", "inner")
        if how != "inner":
            raise AggPlanError("join.how supports only 'inner' (equi-join)")
        out = {"op": op, "left": src("left"), "right": src("right"), "on": pairs, "how": how, "into": into}
    else:
        raise AggPlanError(f"unknown op {op!r}")
    return out


def validate_agg_plan(obj: Any) -> dict:
    """Structural + type-lite validation. Returns a canonicalized plan with valid=True,
    or raises AggPlanError. Table-name dataflow is checked; field existence/types are
    enforced by the reducer against the real artifact data."""
    if not isinstance(obj, dict):
        raise AggPlanError("plan must be a JSON object")
    srcs = obj.get("sources", [])
    if not isinstance(srcs, list) or not srcs or len(srcs) > MAX_SOURCES:
        raise AggPlanError(f"sources must be a non-empty list of <= {MAX_SOURCES}")
    defined, csources = set(), []
    for s in srcs:
        if not isinstance(s, dict) or "artifact_id" not in s or "as" not in s:
            raise AggPlanError(f"source needs artifact_id+as: {s!r}")
        nm = _name(s["as"], "source.as")
        if nm in defined:
            raise AggPlanError(f"duplicate source name {nm!r}")
        defined.add(nm)
        csources.append({"artifact_id": _name(s["artifact_id"], "artifact_id"), "as": nm})

    steps = obj.get("steps", [])
    if not isinstance(steps, list) or not steps or len(steps) > MAX_STEPS:
        raise AggPlanError(f"steps must be a non-empty list of <= {MAX_STEPS}")
    csteps = []
    for s in steps:
        cs = _validate_step(s, defined)
        if cs["into"] in defined:
            raise AggPlanError(f"step writes into an already-defined name {cs['into']!r}")
        defined.add(cs["into"])
        csteps.append(cs)

    out = obj.get("output", {})
    if not isinstance(out, dict) or "from" not in out or "kind" not in out:
        raise AggPlanError("output needs from+kind")
    frm = _name(out["from"], "output.from")
    if frm not in defined:
        raise AggPlanError(f"output.from {frm!r} not defined")
    if out["kind"] not in ("scalar", "table"):
        raise AggPlanError("output.kind must be scalar/table")
    coutput = {"from": frm, "kind": out["kind"]}
    if out["kind"] == "scalar":
        coutput["field"] = _name(out.get("field"), "scalar output.field")
    return {"valid": True, "sources": csources, "steps": csteps, "output": coutput}
