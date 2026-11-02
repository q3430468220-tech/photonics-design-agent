# Contributing

Bug reports from real Lumerical versions, new device templates, new solver backends and clearer documentation are all welcome.

## Development set-up

```bash
git clone https://github.com/q3430468220-tech/photonics-design-agent.git
cd photonics-design-agent
pip install -e .
python -m unittest discover -s tests -t .      # 27 tests, no simulator or API key needed
python examples/demo_scripted.py               # end-to-end on the mock backend
```

## Reporting a Lumerical problem

Run `prism eval <experiment> --manual --backend lumerical` and attach: the full traceback, your Lumerical version, the `sim_result.json` and `eval.json` from the run folder if they were written, and the output of `prism check <experiment>`. The lines most likely to differ between versions are the material names and the mode-expansion result keys in `prism_min/backends/lumerical.py`.

## Adding a device template

A template is a folder under `experiments/` with the five files the loop expects.

1. `program.md` — platform, port interface, FOM, hard constraints, budget, hints. Write it for a new graduate student.
2. `sim_config.py` — `CONFIG = SimConfig(...)`. Choose the noise threshold from two simulations of the same design at different mesh settings.
3. `design.py` — a starting `build()` that passes `prism check`.
4. `model.py` — a crude but honest prior; the agent should want to replace it.
5. `laws.md` — the prior knowledge you are willing to give, tagged `[prior]`.

Then run `prism eval --manual` on the mock *and* on a real backend, confirm the breakdown closes, and add a scripted loop test like `tests/test_loop_mock.py` if the template needs new geometry helpers. If the device is not a 4-port crossing, `prism_min/features.py` and the mock will need a device-specific feature extractor; keep them in a new module rather than branching the crossing one.

## Adding a backend

Subclass `prism_min.backends.base.Backend`, register it with `@register("name")`, and return a `SimResult` whose fields are fractions of injected power per wavelength. Two rules: read every numerical setting from `SimConfig`, never from arguments the agent could influence; and measure `box_total` — do not compute it from the other terms, or the closure check becomes a tautology.

## Style

Plain `unittest`, no required dev dependencies. Keep `numpy`, `matplotlib`, `requests` as the only runtime dependencies unless a backend needs more, in which case import it lazily inside the backend module. Cite the source of every default number in a comment.
