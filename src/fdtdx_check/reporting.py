from __future__ import annotations

from html import escape
import math
from typing import Any


STYLE = """
:root{color-scheme:light dark;--bg:#f4f7fb;--paper:#fff;--ink:#152238;--muted:#607086;
--line:#d9e2ec;--blue:#176b87;--blue-open:#dceef4;--gold:#b7791f;--gold-open:#fff4d6}
@media(prefers-color-scheme:dark){:root{--bg:#0c1422;--paper:#121e30;--ink:#edf4fb;
--muted:#a7b6c8;--line:#2d4058;--blue:#67c1da;--blue-open:#183848;--gold:#e7b85f;--gold-open:#493a19}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif}
main,header{width:min(1080px,calc(100% - 32px));margin:auto}header{padding:52px 0 24px}
.eyebrow{font-size:12px;font-weight:750;letter-spacing:.13em;text-transform:uppercase;color:var(--blue)}
h1{font-size:clamp(30px,5vw,48px);line-height:1.08;margin:8px 0 14px;letter-spacing:-.035em}
h2{font-size:23px;margin:0 0 8px;letter-spacing:-.015em}h3{font-size:16px;margin:0 0 8px}
.lede{max-width:780px;color:var(--muted);font-size:17px}.panel{background:var(--paper);border:1px solid var(--line);
border-radius:16px;padding:24px;margin:0 0 18px;box-shadow:0 8px 28px rgba(20,40,70,.05)}
.summary{border-left:5px solid var(--blue)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}
.metric{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:18px;min-height:116px}
.metric b{display:block;font:750 clamp(22px,3vw,30px)/1.15 ui-monospace,monospace;margin:7px 0}
.label{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;font-weight:700}
.note{color:var(--muted);font-size:13px}.badge{display:inline-block;border:1px solid var(--blue);color:var(--blue);
border-radius:99px;padding:3px 9px;font-size:12px;font-weight:750}.badge.oom{border-style:dashed;border-color:var(--gold);color:var(--gold)}
.badge.fail{border-style:dashed;border-color:var(--gold);color:var(--gold)}
.chart{margin-top:20px}.barrow{display:grid;grid-template-columns:105px minmax(100px,1fr) 112px 54px;gap:10px;
align-items:center;margin:9px 0}.track{height:22px;background:var(--blue-open);border:1px solid var(--line);border-radius:6px;overflow:hidden}
.fill{height:100%;background:var(--blue)}.fill.oom{background:var(--gold-open);border:2px dashed var(--gold)}
.fill.secondary{background:var(--gold)}
.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-variant-numeric:tabular-nums}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:10px 12px;
border-bottom:1px solid var(--line)}th{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.05em}
.tablewrap{overflow:auto}.split{height:42px;display:flex;border:1px solid var(--line);border-radius:8px;overflow:hidden;margin:18px 0 8px}
.split div{display:flex;align-items:center;justify-content:center;min-width:56px;font-weight:750}.out1{background:var(--blue);color:#fff}
.out2{background:var(--gold-open);color:var(--ink);border-left:2px solid var(--paper)}img{display:block;width:100%;border:1px solid var(--line);border-radius:12px}
ul{padding-left:20px}a{color:var(--blue)}footer{color:var(--muted);padding:10px 0 48px;font-size:13px}
@media(max-width:650px){header{padding-top:30px}.panel{padding:18px}.barrow{grid-template-columns:82px 1fr 78px}.barrow .state{display:none}
.compact th,.compact td{padding:8px 4px;font-size:11px}}
"""


def _page(title: str, eyebrow: str, lede: str, body: str) -> str:
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><meta name='color-scheme' content='light dark'>
<title>{escape(title)}</title><style>{STYLE}</style></head><body><header><div class='eyebrow'>{escape(eyebrow)}</div>
<h1>{escape(title)}</h1><p class='lede'>{escape(lede)}</p></header><main>{body}
<footer>Self-contained report. Machine-readable evidence remains beside this file.</footer></main></body></html>"""


def _short(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


def _rasterized_geometry_rows(report: dict[str, Any]) -> str:
    rows = []
    for key, value in report["rasterized_geometry"].items():
        if not isinstance(value, int | float):
            continue
        declared = report["geometry"].get(key)
        declared_text = "--" if declared is None else f"{float(declared):.6g}"
        rows.append(
            f"<tr><td>{escape(str(key).replace('_', ' '))}</td>"
            f"<td class='mono'>{declared_text}</td><td class='mono'>{float(value):.6g}</td></tr>"
        )
    return "".join(rows)


def profile_html(report: dict[str, Any]) -> str:
    attempts = report["attempts"]
    cell_values = [max(int(row.get("cells_total") or 1), 1) for row in attempts]
    logs = [math.log10(value) for value in cell_values]
    lo, hi = min(logs), max(logs)
    span = max(hi - lo, 1e-12)
    rows, bars = [], []
    for row, value, log_value in zip(attempts, cell_values, logs, strict=True):
        status = str(row["status"])
        is_oom = row.get("failure_kind") == "OOM"
        state = "OOM" if is_oom else status
        width = 12 + 88 * (log_value - lo) / span
        elapsed = row.get("elapsed_seconds")
        elapsed_text = "--" if elapsed is None else f"{float(elapsed):.3f}"
        cls = " oom" if is_oom else ""
        label = f"{row['devices']}d / {row['case']}"
        bars.append(
            f"<div class='barrow'><span class='mono'>{escape(label)}</span><div class='track' title='{value:,} cells'>"
            f"<div class='fill{cls}' style='width:{width:.2f}%'></div></div><span class='mono'>{value:,}</span>"
            f"<span class='state badge{cls}'>{escape(state)}</span></div>"
        )
        rows.append(
            f"<tr><td class='mono'>{escape(label)}</td><td>{state}</td><td class='mono'>{value:,}</td>"
            f"<td class='mono'>{int(row.get('cells_per_device') or value):,}</td><td class='mono'>{elapsed_text}</td></tr>"
        )
    bounded = bool(report["capacity_boundary_observed"])
    boundary = "OOM boundary observed" if bounded else "Tested lower bound"
    interpretation = (
        "The largest passing case is bracketed by a following allocation OOM."
        if bounded
        else "Every configured case passed; the true allocation ceiling is higher than this run."
    )
    kinds = ", ".join(escape(str(v)) for v in report["device_kinds"])
    body = f"""
<section class='panel summary'><h2>Technical summary</h2><p><strong>{escape(boundary)}.</strong>
The largest successful FP64 allocation was <span class='mono'>{int(report['largest_tested_cells']):,}</span> Yee cells.
The science planner is capped at <span class='mono'>{int(report['recommended_safe_cells']):,}</span> cells
({float(report['safe_fraction'])*100:.0f}% of that observation). {escape(interpretation)}</p></section>
<section class='grid' aria-label='Capacity metrics'>
<div class='metric'><span class='label'>Largest pass</span><b>{_short(int(report['largest_tested_cells']))}</b><span class='note'>Yee cells</span></div>
<div class='metric'><span class='label'>Science safe cap</span><b>{_short(int(report['recommended_safe_cells']))}</b><span class='note'>{float(report['safe_fraction'])*100:.0f}% policy</span></div>
<div class='metric'><span class='label'>Logical devices</span><b>{int(report['logical_devices_available'])}</b><span class='note'>{kinds}</span></div>
<div class='metric'><span class='label'>Numerics</span><b>{escape(str(report['precision']).upper())}</b><span class='note'>{escape(str(report['backend']).upper())} backend</span></div>
</section>
<section class='panel'><h2>Allocated Yee cells by attempt</h2><p>Each row is a fresh FDTDX process. Bar length uses a log10 scale so small and large allocations remain visible; exact cell counts are printed at right.</p>
<div class='chart'>{''.join(bars)}</div></section>
<section class='panel'><h2>Exact attempt record</h2><p>The table preserves the audit values behind the visual. A non-OOM failure invalidates the profiling run instead of being treated as a capacity boundary.</p>
<div class='tablewrap'><table><thead><tr><th>Device / case</th><th>State</th><th>Total cells</th><th>Cells / device</th><th>Seconds</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div></section>
<section class='panel'><h2>Scope and interpretation</h2><ul>
<li>Hardware pool: <span class='mono'>{escape(str(report['hardware_id']))}</span>.</li>
<li>Declared physical accelerators: {report.get('expected_physical_accelerators', '--')}; declared aggregate HBM: {report.get('declared_aggregate_hbm_gib', '--')} GiB.</li>
<li>The profile measures FDTDX allocation and short-step execution, not long-run stability or converged device accuracy.</li>
<li>The next action is to inspect <a href='../capacity.yaml'>capacity.yaml</a>, then run the capacity-matched science stage.</li>
</ul></section>"""
    return _page(
        "FDTDX FP64 capacity profile",
        "Stage 1 / empirical capacity",
        "A device-count and grid-size sweep that separates a measured allocation boundary from a tested lower bound.",
        body,
    )


def science_html(report: dict[str, Any]) -> str:
    m = report["metrics"]
    total = float(m["total_transmission"])
    share1 = 50.0 if total <= 0 else 100 * float(m["T1"]) / total
    share2 = 100 - share1
    geometry = report["geometry"]
    geometry_rows = "".join(
        f"<tr><td>{escape(str(key).replace('_', ' '))}</td><td class='mono'>{escape(str(value))}</td></tr>"
        for key, value in geometry.items()
    )
    rasterized_rows = _rasterized_geometry_rows(report)
    validation_rows = "".join(
        f"<tr><td>{escape(name.replace('_', ' '))}</td><td><span class='badge{' ' if passed else ' fail'}'>"
        f"{'PASS' if passed else 'FAIL'}</span></td></tr>"
        for name, passed in report["validation"]["checks"].items()
    )
    status_text = "All configured sanity checks passed" if report["measurement_valid"] else "One or more sanity checks failed"
    body = f"""
<section class='panel summary'><h2>Technical summary</h2><p><strong>{escape(status_text)}.</strong>
The FP64 run completed {int(report['steps']):,} Maxwell steps on {int(report['logical_devices'])} logical device(s)
with {int(report['cells_total']):,} Yee cells. The displayed overlap values verify the end-to-end source, field,
detector, and reporting path; they are not a converged component specification.</p></section>
<section class='grid' aria-label='Science metrics'>
<div class='metric'><span class='label'>Total modal overlap</span><b>{total:.4f}</b><span class='note'>T1 + T2, diagnostic normalization</span></div>
<div class='metric'><span class='label'>Diagnostic phase</span><b>{float(m['relative_phase_deg']):.2f} deg</b><span class='note'>mode-basis gauge dependent</span></div>
<div class='metric'><span class='label'>Imbalance</span><b>{float(m['imbalance_db']):.2f} dB</b><span class='note'>10 log10(T1 / T2)</span></div>
<div class='metric'><span class='label'>Detector accumulation</span><b>{escape(str(report['detector_dtype']))}</b><span class='note'>{escape(str(report['precision']))} propagation</span></div>
</section>
<section class='panel'><h2>Electric-field propagation through the generic MMI</h2><p>The normalized real Ey phasor shows whether the launched TE-like mode reaches the multimode section and both output waveguides. Color encodes field sign as well as magnitude.</p>
<img src='field.png' alt='Normalized real Ey field through a generic one-by-two MMI'></section>
<section class='panel'><h2>Output-mode composition</h2><p>This stacked bar compares the two detected output powers within T1 + T2. It does not treat uncollected radiation as a third calibrated channel.</p>
<div class='split' role='img' aria-label='Output one {share1:.1f} percent; output two {share2:.1f} percent'>
<div class='out1' style='width:{share1:.3f}%'>T1 {share1:.1f}%</div><div class='out2' style='width:{share2:.3f}%'>T2 {share2:.1f}%</div></div>
<div class='tablewrap'><table class='compact'><thead><tr><th>Port</th><th>Real overlap</th><th>Imag overlap</th><th>Power</th></tr></thead><tbody>
<tr><td>S21</td><td class='mono'>{float(m['S21_real']):.7g}</td><td class='mono'>{float(m['S21_imag']):.7g}</td><td class='mono'>{float(m['T1']):.7g}</td></tr>
<tr><td>S31</td><td class='mono'>{float(m['S31_real']):.7g}</td><td class='mono'>{float(m['S31_imag']):.7g}</td><td class='mono'>{float(m['T2']):.7g}</td></tr>
</tbody></table></div></section>
<section class='panel'><h2>Per-run physics sanity checks</h2><p>Each gate is reported independently so a nonzero but nonphysical result cannot pass silently.</p>
<div class='tablewrap'><table><thead><tr><th>Check</th><th>Result</th></tr></thead><tbody>{validation_rows}</tbody></table></div></section>
<section class='panel'><h2>Model and run definition</h2><div class='grid'>
<div><h3>Execution</h3><table><tbody><tr><td>Backend</td><td class='mono'>{escape(str(report['backend']))}</td></tr>
	<tr><td>Hardware pool</td><td class='mono'>{escape(str(report['hardware_id']))}</td></tr>
	<tr><td>Precision</td><td class='mono'>{escape(str(report['precision']))}</td></tr><tr><td>Steps</td><td class='mono'>{int(report['steps']):,}</td></tr>
	<tr><td>Field dtypes</td><td class='mono'>{escape(str(report['field_state_dtypes']))}</td></tr>
	<tr><td>Detector dtype</td><td class='mono'>{escape(str(report['detector_dtype']))}</td></tr><tr><td>Time</td><td class='mono'>{float(report['time_fs']):g} fs</td></tr>
	<tr><td>PML</td><td class='mono'>{int(report['pml_cells'])} cells / {float(report['pml_actual_um']):.3f} um</td></tr>
	<tr><td>Grid</td><td class='mono'>{int(report['resolution_nm'])} nm / {' x '.join(str(v) for v in report['grid_shape'])}</td></tr>
	<tr><td>Grid contract</td><td class='mono'>{escape(str(report['grid_contract_hash'])[:12])}</td></tr>
	<tr><td>Output extension</td><td class='mono'>{int(report['output_extension_cells'])} cells / {float(report['output_extension_um']):.3f} um</td></tr>
<tr><td>Elapsed</td><td class='mono'>{float(report['elapsed_seconds']):.3f} s</td></tr><tr><td>Field shards</td><td class='mono'>{int(report['field_shards'])}</td></tr></tbody></table></div>
<div><h3>Declared geometry</h3><table><tbody>{geometry_rows}</tbody></table></div></div>
<h3>Grid-realized geometry</h3><div class='tablewrap'><table><thead><tr><th>Quantity</th><th>Declared um</th><th>Realized um</th></tr></thead><tbody>{rasterized_rows}</tbody></table></div></section>
<section class='panel'><h2>Limitations and next check</h2><ul>
<li>The geometry is synthetic and contains no imported GDS or foundry PCell.</li>
<li>These overlap values are descriptive diagnostics, not qualified insertion loss or production S-parameters.</li>
<li>The displayed relative phase is diagnostic because independently solved port modes may carry arbitrary global phase.</li>
<li>Time convergence is evaluated by the suite report; grid, port, and material-model convergence remain separate requirements.</li>
<li>Review <a href='report.json'>report.json</a> for exact per-run evidence; the suite report records cross-window convergence and capacity selection.</li>
</ul></section>"""
    return _page(
        "Generic 1x2 MMI FP64 science check",
        "Stage 2 / capacity-matched physics",
        "A synthetic, non-confidential mode-source and complex-output test sized from the measured accelerator capacity.",
        body,
    )


def science_suite_html(report: dict[str, Any]) -> str:
    cases = report["cases"]
    convergence = report["convergence"]
    final = cases[-1]
    max_magnitude = max(
        math.hypot(float(case["metrics"][f"{port}_real"]), float(case["metrics"][f"{port}_imag"]))
        for case in cases
        for port in ("S21", "S31")
    )
    bar_rows, table_rows = [], []
    for case in cases:
        m = case["metrics"]
        s21 = math.hypot(float(m["S21_real"]), float(m["S21_imag"]))
        s31 = math.hypot(float(m["S31_real"]), float(m["S31_imag"]))
        for port, magnitude, cls in (("S21", s21, ""), ("S31", s31, " secondary")):
            width = 100 * magnitude / max(max_magnitude, 1e-30)
            bar_rows.append(
                f"<div class='barrow'><span class='mono'>{float(case['time_fs']):g} fs {port}</span>"
                f"<div class='track'><div class='fill{cls}' style='width:{width:.3f}%'></div></div>"
                f"<span class='mono'>{magnitude:.6f}</span><span class='state'></span></div>"
            )
        table_rows.append(
            f"<tr><td class='mono'>{float(case['time_fs']):g}</td><td class='mono'>{int(case['logical_devices'])}</td>"
            f"<td>{escape(str(case['status']))}</td>"
            f"<td class='mono'>{s21:.7g}</td><td class='mono'>{s31:.7g}</td>"
            f"<td class='mono'>{float(m['total_transmission']):.7g}</td><td class='mono'>{float(m['imbalance_db']):.4f}</td>"
            f"<td><a href='{escape(str(case['report_path']))}'>case report</a></td></tr>"
        )
    checks = report["validation"]
    validation_rows = "".join(
        f"<tr><td>{escape(name.replace('_', ' '))}</td><td><span class='badge{' ' if passed else ' fail'}'>"
        f"{'PASS' if passed else 'FAIL'}</span></td></tr>"
        for name, passed in checks.items()
        if name != "all_checks_pass"
    )
    overall = "The FP64 physics regression passed" if report["status"] == "PASS" else "The FP64 physics regression failed"
    resolution_rows = []
    for row in report["resolution_attempts"]:
        attempt_text = "; ".join(
            f"{int(item['logical_devices'])}d {int(item['completed_time_samples'])}/{len(report['cases'])} "
            f"{item['status'] if item['status'] == 'PASS_EXECUTION' else item.get('failure_kind') or 'FAIL'}"
            for item in row["device_attempts"]
        )
        selected = row.get("selected_logical_devices")
        resolution_rows.append(
            f"<tr><td class='mono'>{int(row['resolution_nm'])}</td><td>{escape(str(row['status']))}</td>"
            f"<td class='mono'>{int(row['planned_cells']):,}</td><td class='mono'>{int(row['required_logical_devices'])}</td>"
            f"<td class='mono'>{'--' if selected is None else int(selected)}</td>"
            f"<td>{escape(attempt_text)}</td></tr>"
        )
    rasterized_rows = _rasterized_geometry_rows(report)
    body = f"""
<section class='panel summary'><h2>Technical summary</h2><p><strong>{escape(overall)}.</strong>
	All detector accumulators used <span class='mono'>{escape(str(report['detector_dtype']))}</span> and the physical PML thickness
	remained {float(report['pml_actual_um']):.3f} um. The last-two-window maximum output-magnitude drift was
	{float(convergence['maximum_observed_drift'])*100:.3f}% against a {float(convergence['maximum_allowed_drift'])*100:.2f}% limit.
	All windows used one grid contract and one fixed logical-device count.</p></section>
<section class='grid' aria-label='Regression metrics'>
<div class='metric'><span class='label'>Suite status</span><b>{escape(str(report['status']))}</b><span class='note'>sanity plus time convergence</span></div>
<div class='metric'><span class='label'>Detector dtype</span><b>{escape(str(report['detector_dtype']))}</b><span class='note'>{escape(str(report['precision']))} propagation</span></div>
	<div class='metric'><span class='label'>Grid</span><b>{int(report['resolution_nm'])} nm</b><span class='note'>{int(report['cells_total']):,} Yee cells</span></div>
	<div class='metric'><span class='label'>Logical devices</span><b>{int(report['logical_devices'])}</b><span class='note'>fixed across all windows</span></div>
	<div class='metric'><span class='label'>Time drift</span><b>{float(convergence['maximum_observed_drift'])*100:.3f}%</b><span class='note'>{float(convergence['comparison_fs'][0]):g} to {float(convergence['comparison_fs'][1]):g} fs</span></div>
</section>
<section class='panel'><h2>Output magnitudes by time window</h2><p>Discrete bars compare |S21| and |S31| at each configured stop time. The convergence gate uses only the last two windows.</p>
<div class='chart'>{''.join(bar_rows)}</div></section>
	<section class='panel'><h2>Capacity-aware resolution selection</h2><p>The planner starts at the finest profile-safe candidate. An allocation OOM moves to the next declared device milestone and reruns every time window on the same canonical grid.</p>
	<div class='tablewrap'><table><thead><tr><th>Resolution nm</th><th>Execution</th><th>Planned cells</th><th>Required devices</th><th>Selected</th><th>Milestone attempts</th></tr></thead><tbody>{''.join(resolution_rows)}</tbody></table></div></section>
	<section class='panel'><h2>Exact time-window results</h2><div class='tablewrap'><table><thead><tr><th>Time fs</th><th>Devices</th><th>Status</th><th>|S21|</th><th>|S31|</th><th>T1 + T2</th><th>Imbalance dB</th><th>Evidence</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table></div></section>
<section class='panel'><h2>Declared versus grid-realized geometry</h2><p>Grid alignment deliberately preserves the mirror plane. Values below expose any discretization change instead of silently treating the continuous dimensions as exact.</p>
<div class='tablewrap'><table><thead><tr><th>Quantity</th><th>Declared um</th><th>Realized um</th></tr></thead><tbody>{rasterized_rows}</tbody></table></div></section>
<section class='panel'><h2>Latest electric-field phasor</h2><p>The final time window supplies the displayed field; earlier fields remain linked from their case reports.</p>
<img src='{escape(str(final['field_path']))}' alt='Latest normalized real Ey field through the generic MMI'></section>
<section class='panel'><h2>Validation gates</h2><div class='tablewrap'><table><thead><tr><th>Gate</th><th>Result</th></tr></thead><tbody>{validation_rows}</tbody></table></div></section>
	<section class='panel'><h2>Interpretation limits</h2><ul>
	<li>Hardware pool: <span class='mono'>{escape(str(report['hardware_id']))}</span>.</li>
	<li>Grid contract <span class='mono'>{escape(str(report['grid_contract_hash'])[:12])}</span> is identical across all time windows; the output waveguides include {float(report['output_extension_um']):.3f} um of straight padding to contact the positive-x PML.</li>
<li>Time convergence is based on output magnitudes, which are invariant to the arbitrary phase gauge of independently solved port modes.</li>
<li>The diagnostic relative phase is retained in each case report but is not a PASS criterion.</li>
<li>Constant-index synthetic geometry is an accelerator regression, not a foundry-calibrated component model.</li>
<li>Grid-resolution and port-plane convergence remain required before claiming quantitative S-parameters.</li>
</ul></section>"""
    return _page(
        "FDTDX FP64 physics regression",
        "Stage 2 / sanity and time convergence",
        "A capacity-matched, non-confidential validation of detector precision, passivity, symmetry, and output stability.",
        body,
    )
