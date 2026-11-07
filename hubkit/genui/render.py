"""Streamlit renderer for agentic UI specs (AgenticGenUI protocol: ``{"type", "props"}``).

``REGISTRY`` maps a component ``type`` to a function returning HTML, like AgenticGenUI's
``componentRegistry``; unknown types fall back to their raw JSON, as its ``AgentRenderer``
falls back to raw text. Components animate with CSS only (no scripts): items created by
the latest agent event (``seq``) enter with a fade or slide, the running step pulses,
running lenses spin, retries shake, and the progress bar grows from its previous value.

Theming: neutrals derive from the inherited text colour (``color-mix`` with ``currentColor``), so the
panel fits Streamlit's light and dark themes. Accents (violet primary, cyan accent, status colours)
switch with ``Renderer(theme="dark" | "light")`` (from ``st.context.theme.type``) or, when the theme is
unknown, with the browser's ``prefers-color-scheme``. The palette matches ``.streamlit/config.toml``.
"""

from __future__ import annotations

import html
import json
from typing import Callable

CSS = """
<style>
.ga{
  /* neutrals derive from the inherited text colour, so they fit Streamlit's light and dark themes */
  --ga-surface:color-mix(in srgb,currentColor 3.5%,transparent);
  --ga-border:color-mix(in srgb,currentColor 13%,transparent);
  --ga-track:color-mix(in srgb,currentColor 10%,transparent);
  --ga-muted:color-mix(in srgb,currentColor 64%,transparent);
  --ga-primary:#6D4AFF;--ga-accent:#0EA5E9;--ga-ok:#16A34A;--ga-warn:#D97706;--ga-bad:#E11D48;
  --ga-glow:rgba(109,74,255,.45);--ga-shadow:0 1px 2px rgba(16,18,27,.05),0 10px 28px -14px rgba(16,18,27,.18);
  --ga-on-primary:#fff;
  font-family:inherit;display:flex;flex-direction:column;gap:12px;margin:4px 0 8px
}
.ga.ga-light{--ga-surface:#FFFFFF}  /* known light theme: white cards on the off-white page */
.ga.ga-dark{--ga-primary:#8B6CFF;--ga-accent:#22D3EE;--ga-ok:#22C55E;--ga-warn:#F59E0B;--ga-bad:#FB7185;
  --ga-glow:rgba(139,108,255,.55);--ga-shadow:0 1px 2px rgba(0,0,0,.35),0 14px 34px -18px rgba(0,0,0,.7)}
@media (prefers-color-scheme:dark){.ga:not(.ga-light):not(.ga-dark){--ga-primary:#8B6CFF;--ga-accent:#22D3EE;
  --ga-ok:#22C55E;--ga-warn:#F59E0B;--ga-bad:#FB7185;--ga-glow:rgba(139,108,255,.55);
  --ga-shadow:0 1px 2px rgba(0,0,0,.35),0 14px 34px -18px rgba(0,0,0,.7)}}
.ga *{box-sizing:border-box}
.ga-card{border:1px solid var(--ga-border);border-radius:16px;padding:14px 16px;background:var(--ga-surface);box-shadow:var(--ga-shadow)}
.ga-enter{animation:ga-in .45s ease-out both}
@keyframes ga-in{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.ga-pbar{height:8px;border-radius:99px;background:var(--ga-track);overflow:hidden;position:relative}
.ga-pfill{height:100%;border-radius:99px;background:linear-gradient(90deg,var(--ga-primary),var(--ga-accent));animation:ga-grow .6s ease-out both;width:var(--to)}
.ga-pfill.success{background:linear-gradient(90deg,var(--ga-ok),color-mix(in srgb,var(--ga-ok) 55%,var(--ga-accent)))}
.ga-pfill.error{background:var(--ga-bad)}
@keyframes ga-grow{from{width:var(--from)}to{width:var(--to)}}
.ga-pbar.working::after{content:"";position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(255,255,255,.4),transparent);animation:ga-shim 1.4s infinite}
@keyframes ga-shim{from{transform:translateX(-100%)}to{transform:translateX(100%)}}
.ga-row{display:flex;justify-content:space-between;align-items:baseline;gap:8px;font-size:.86rem;margin-bottom:8px}
.ga-muted{color:var(--ga-muted);font-size:.78rem}
.ga-tl{list-style:none;margin:0;padding:0}
.ga-tl li{display:flex;gap:10px;align-items:flex-start;position:relative;padding:0 0 10px 0}
.ga-tl li:not(:last-child)::before{content:"";position:absolute;left:8px;top:20px;bottom:2px;width:2px;border-radius:2px;background:var(--ga-border)}
.ga-tl li.completed:not(:last-child)::before{background:linear-gradient(var(--ga-primary),var(--ga-accent))}
.ga-dot{flex:0 0 18px;height:18px;border-radius:50%;border:2px solid var(--ga-border);display:flex;align-items:center;justify-content:center;font-size:11px;line-height:1;margin-top:1px}
.completed .ga-dot{background:var(--ga-primary);border-color:var(--ga-primary);color:var(--ga-on-primary);animation:ga-pop .35s ease-out}
.in-progress .ga-dot{border-color:var(--ga-primary);animation:ga-pulse 1.3s infinite}
.waiting .ga-dot{border-color:var(--ga-warn);background:color-mix(in srgb,var(--ga-warn) 18%,transparent);animation:ga-pulse-amber 1.6s infinite}
.skipped .ga-dot{border-style:dashed}
.skipped .ga-tlt,.pending .ga-tlt{opacity:.55}
@keyframes ga-pop{0%{transform:scale(.4)}70%{transform:scale(1.2)}100%{transform:scale(1)}}
@keyframes ga-pulse{0%{box-shadow:0 0 0 0 var(--ga-glow)}100%{box-shadow:0 0 0 9px transparent}}
@keyframes ga-pulse-amber{0%{box-shadow:0 0 0 0 color-mix(in srgb,var(--ga-warn) 55%,transparent)}100%{box-shadow:0 0 0 9px transparent}}
.ga-tlt{font-size:.88rem}.in-progress .ga-tlt{font-weight:600;color:var(--ga-primary)}
.ga-cl{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:8px;margin-top:4px}
.ga-item{border:1px solid var(--ga-border);border-radius:12px;padding:9px 11px;font-size:.84rem;display:flex;gap:8px;align-items:flex-start;background:var(--ga-surface)}
.ga-item .ga-ic{flex:0 0 16px;height:16px;margin-top:2px;text-align:center}
.ga-item.running{border-color:color-mix(in srgb,var(--ga-primary) 45%,transparent);background:color-mix(in srgb,var(--ga-primary) 6%,transparent)}
.ga-spin{border:2px solid color-mix(in srgb,var(--ga-primary) 25%,transparent);border-top-color:var(--ga-primary);border-radius:50%;animation:ga-rot .8s linear infinite}
@keyframes ga-rot{to{transform:rotate(360deg)}}
.ga-item.done{border-color:color-mix(in srgb,var(--ga-ok) 45%,transparent)}.ga-item.done .ga-ic{color:var(--ga-ok)}
.ga-item.just-done{animation:ga-flash .9s ease-out}
@keyframes ga-flash{from{background:color-mix(in srgb,var(--ga-ok) 22%,transparent)}to{background:var(--ga-surface)}}
.ga-item.retrying{border-color:color-mix(in srgb,var(--ga-warn) 60%,transparent);animation:ga-shake .5s ease-in-out}.ga-item.retrying .ga-ic{color:var(--ga-warn)}
@keyframes ga-shake{0%,100%{transform:none}25%{transform:translateX(-3px)}75%{transform:translateX(3px)}}
.ga-item.skipped{opacity:.6;border-style:dashed}
.ga-item.error{border-color:color-mix(in srgb,var(--ga-bad) 60%,transparent)}.ga-item.error .ga-ic{color:var(--ga-bad)}
.ga-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.ga-metric{border:1px solid var(--ga-border);border-radius:14px;padding:12px 13px 11px;background:var(--ga-surface);box-shadow:var(--ga-shadow);position:relative;overflow:hidden}
.ga-metric::before{content:"";position:absolute;left:0;top:0;right:0;height:3px;background:linear-gradient(90deg,var(--ga-primary),var(--ga-accent));opacity:.85}
.ga-metric.ga-enter{animation:ga-flip .55s ease-out both}
@keyframes ga-flip{from{opacity:0;transform:perspective(400px) rotateX(-35deg)}to{opacity:1;transform:none}}
.ga-mv{font-size:1.3rem;font-weight:700;margin:2px 0;letter-spacing:-.01em}
.ga-up{color:var(--ga-ok)}.ga-down{color:var(--ga-bad)}
.ga-banner{border:1px solid color-mix(in srgb,var(--ga-primary) 30%,transparent);border-left:4px solid var(--ga-primary);border-radius:12px;padding:9px 13px;
  background:color-mix(in srgb,var(--ga-primary) 8%,transparent);font-size:.86rem}
.ga-banner.success{border-color:color-mix(in srgb,var(--ga-ok) 35%,transparent);border-left-color:var(--ga-ok);background:color-mix(in srgb,var(--ga-ok) 8%,transparent)}
.ga-banner.warning{border-color:color-mix(in srgb,var(--ga-warn) 35%,transparent);border-left-color:var(--ga-warn);background:color-mix(in srgb,var(--ga-warn) 9%,transparent)}
.ga-banner.error{border-color:color-mix(in srgb,var(--ga-bad) 35%,transparent);border-left-color:var(--ga-bad);background:color-mix(in srgb,var(--ga-bad) 9%,transparent)}
.ga-banner.muted{border-color:var(--ga-border);border-left-color:var(--ga-muted);background:var(--ga-surface)}
.ga-toasts{display:flex;flex-direction:column;gap:6px;align-items:flex-end}
.ga-toast{max-width:560px;border-radius:12px;padding:8px 13px;font-size:.82rem;background:color-mix(in srgb,var(--ga-warn) 12%,transparent);
  border:1px solid color-mix(in srgb,var(--ga-warn) 45%,transparent);box-shadow:var(--ga-shadow)}
.ga-toast.error{background:color-mix(in srgb,var(--ga-bad) 10%,transparent);border-color:color-mix(in srgb,var(--ga-bad) 50%,transparent)}
.ga-toast.new{animation:ga-slide .45s ease-out both}
.ga-toast.old{opacity:.6}
@keyframes ga-slide{from{opacity:0;transform:translateX(24px)}to{opacity:1;transform:none}}
.ga-raw{font-family:ui-monospace,Consolas,monospace;font-size:.75rem;white-space:pre-wrap;color:var(--ga-muted)}
@media (prefers-reduced-motion:reduce){.ga *{animation:none!important}}
</style>
"""

THEMES = ("light", "dark")
ICONS = {"done": "✓", "skipped": "–", "error": "!", "pending": "○", "retrying": "↻"}


def e(x) -> str:
    return html.escape(str(x if x is not None else ""))


class Renderer:
    """Holds state between redraws (previous progress) so animations continue smoothly."""

    def __init__(self, theme: str | None = None) -> None:
        self.prev_progress = 0
        self.theme = theme if theme in THEMES else None  # None: follow prefers-color-scheme

    def render(self, specs: list[dict], current_seq: int | None = None) -> str:
        seq = current_seq if current_seq is not None else max((s.get("props", {}).get("seq", 0) for s in specs), default=0)
        parts = [REGISTRY.get(s.get("type"), _raw)(self, s.get("props") or {}, seq) for s in specs]
        cls = f"ga ga-{self.theme}" if self.theme else "ga"
        return CSS + f'<div class="{cls}">' + "".join(parts) + "</div>"


def _enter(props: dict, seq: int) -> str:
    return " ga-enter" if props.get("seq") == seq else ""


def _progress(r: Renderer, p: dict, seq: int) -> str:
    value = int(p.get("value", 0))
    frm, r.prev_progress = r.prev_progress, value
    working = p.get("variant") == "default"
    return (f'<div class="ga-card"><div class="ga-row"><b>{e(p.get("label"))}</b><span class="ga-muted">{e(p.get("detail"))}</span></div>'
            f'<div class="ga-pbar{" working" if working else ""}"><div class="ga-pfill {e(p.get("variant", ""))}" '
            f'style="--from:{frm}%;--to:{value}%"></div></div></div>')


def _timeline(r: Renderer, p: dict, seq: int) -> str:
    rows = []
    for ev in p.get("events", []):
        status = ev.get("status", "pending")
        mark = "✓" if status == "completed" else ""
        detail = f'<div class="ga-muted">{e(ev.get("detail"))}</div>' if ev.get("detail") else ""
        rows.append(f'<li class="{e(status)}"><span class="ga-dot">{mark}</span><div><div class="ga-tlt">{e(ev.get("title"))}</div>{detail}</div></li>')
    return f'<div class="ga-card"><ul class="ga-tl">{"".join(rows)}</ul></div>'


def _checklist(r: Renderer, p: dict, seq: int) -> str:
    items = p.get("items", [])
    cells = []
    for it in items:
        st = it.get("status", "pending")
        icon = '<span class="ga-ic ga-spin"></span>' if st == "running" else f'<span class="ga-ic">{ICONS.get(st, "○")}</span>'
        just = " just-done" if st == "done" and it.get("seq") == seq else ""
        detail = f'<div class="ga-muted">{e(it.get("detail"))}</div>' if it.get("detail") else ""
        cells.append(f'<div class="ga-item {e(st)}{just}">{icon}<div><div>{e(it.get("label"))}</div>{detail}</div></div>')
    return (f'<div class="ga-card"><div class="ga-row"><b>{e(p.get("title"))}</b>'
            f'<span class="ga-muted">{int(p.get("completed", 0))}/{len(items)}</span></div><div class="ga-cl">{"".join(cells)}</div></div>')


def _metric(r: Renderer, p: dict, seq: int) -> str:
    kind = p.get("changeType")
    arrow = {"increase": '<span class="ga-up">▲</span> ', "decrease": '<span class="ga-down">▼</span> '}.get(kind, "")
    change = f'<div class="ga-muted">{arrow}{e(p.get("change"))}</div>' if p.get("change") else ""
    return (f'<div class="ga-metric{_enter(p, seq)}" title="{e(p.get("formula"))}"><div class="ga-muted">{e(p.get("title"))}</div>'
            f'<div class="ga-mv">{e(p.get("value"))}</div>{change}<div class="ga-muted">{e(p.get("description"))}</div></div>')


def _metric_grid(r: Renderer, p: dict, seq: int) -> str:
    return '<div class="ga-grid">' + "".join(_metric(r, c.get("props", {}), seq) for c in p.get("cards", [])) + "</div>"


def _banner(r: Renderer, p: dict, seq: int) -> str:
    desc = f'<div class="ga-muted">{e(p.get("description"))}</div>' if p.get("description") else ""
    return f'<div class="ga-banner {e(p.get("variant", "info"))}{_enter(p, seq)}"><b>{e(p.get("title"))}</b>{desc}</div>'


def _toasts(r: Renderer, p: dict, seq: int) -> str:
    cur = p.get("currentSeq", seq)
    rows = [f'<div class="ga-toast {e(t.get("variant", ""))} {"new" if t.get("seq") == cur else "old"}">{e(t.get("message"))}</div>'
            for t in p.get("toasts", [])]
    return '<div class="ga-toasts">' + "".join(rows) + "</div>"


def _raw(r: Renderer, p: dict, seq: int) -> str:
    return f'<div class="ga-card ga-raw">{e(json.dumps(p, ensure_ascii=False)[:2000])}</div>'


REGISTRY: dict[str, Callable[[Renderer, dict, int], str]] = {
    "ProgressBar": _progress,
    "Timeline": _timeline,
    "ChecklistWithProgress": _checklist,
    "MetricCard": _metric,
    "MetricCardGrid": _metric_grid,
    "InfoBanner": _banner,
    "ToastStack": _toasts,
}
