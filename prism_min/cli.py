"""Command-line interface.

    prism init  <dir> [--template soi_crossing]   create a new experiment from a template
    prism check <dir>                             build design.py, run DRC + sanity, print features
    prism eval  <dir> --manual [--backend mock]   one human-triggered simulation (guard bypassed)
    prism run   <dir> [--iters N] [--sims N]      run the agent loop
    prism status <dir>                            journal summary
    prism plot   <dir> [--log] [--all-labels]     evolution chart (also refreshed after every iteration)
    prism review <dir>                            fresh-session audit of laws.md + journal

.env in the current directory or the repo root is loaded if present.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    for candidate in (Path.cwd() / ".env", PKG_ROOT / ".env"):
        if candidate.exists():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            break


def cmd_init(a):
    src = PKG_ROOT / "experiments" / a.template
    if not src.exists():
        sys.exit(f"template {a.template!r} not found in {src.parent}")
    dst = Path(a.dir)
    if dst.exists() and any(dst.iterdir()):
        sys.exit(f"{dst} exists and is not empty")
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("program.md", "sim_config.py", "design.py", "model.py", "laws.md"):
        shutil.copy(src / name, dst / name)
    print(f"created experiment at {dst} from template {a.template}")
    print("next: edit program.md, then `prism check` and `prism eval --manual` to validate the judge")


def cmd_check(a):
    from .checks import format_violations, run_drc
    from .experiment import DesignError, Experiment
    from .families import family_for
    from .features import format_features

    exp = Experiment(a.dir)
    try:
        geom = exp.build_design()
    except DesignError as e:
        sys.exit(f"design.py: {e}")
    cfg = exp.config
    fam = family_for(cfg)
    print(f"family {fam.name}  design hash {geom.hash()}  config hash {cfg.hash()}  bbox {tuple(round(v, 3) for v in geom.bbox())}")
    try:
        print("features:", format_features(fam.features(geom, cfg)))
    except Exception as e:  # noqa: BLE001
        print(f"features: (error: {e})")
    print("DRC:    ", format_violations(run_drc(geom, cfg)))
    print("checks: ", format_violations(fam.check(geom, cfg)))
    try:
        p = exp.predict(geom)
        print(f"model.py predicts FOM {p['fom']:.4f} +- {p['sigma']:.4f} dB  ({p.get('notes','')})")
    except DesignError as e:
        print(f"model.py: {e}")
    if a.plot:
        from .viz import plot_geometry

        out = plot_geometry(geom, Path(a.dir) / "design_preview.png", title="design.py")
        print("preview:", out)


def cmd_eval(a):
    from .backends import get_backend
    from .evaluate import evaluate
    from .experiment import Experiment
    from .journal import Journal

    from .families import family_for

    exp = Experiment(a.dir)
    backend = get_backend(a.backend or _default_backend(exp, a.real))
    try:
        it = a.iteration if a.iteration is not None else Journal(exp.dir).next_iteration()
        res = evaluate(exp, backend, it, manual=True)
    finally:
        backend.close()
    print(res.short())
    if res.breakdown:
        from .agent import _table_from_dict

        print(_table_from_dict(res.breakdown, exp.config))
    if res.checks:
        print("checks:", "; ".join(res.checks))
    print("run dir:", res.run_dir)
    if not res.valid:
        sys.exit(1)


def cmd_run(a):
    from .agent import Agent
    from .backends import get_backend
    from .experiment import Experiment
    from .llm import get_llm

    exp = Experiment(a.dir)
    a.backend = a.backend or _default_backend(exp, a.real)
    provider = (a.provider or os.environ.get("PRISM_LLM_PROVIDER", "anthropic")).lower()
    if provider == "scripted":
        sys.exit("the scripted provider is for tests; run examples/demo_scripted.py for an offline demonstration")
    backend = get_backend(a.backend)
    llm = get_llm(provider, model=a.model) if a.model else get_llm(provider)
    print(f"experiment {exp.dir}\nbackend {a.backend} | llm {llm.name} {getattr(llm, 'model', '')}")
    agent = Agent(exp, backend, llm)
    try:
        agent.run(max_iters=a.iters, max_sims=a.sims)
    except KeyboardInterrupt:
        print("\ninterrupted - journal is consistent up to the last completed iteration")
    finally:
        backend.close()
    cmd_status(a)


def _default_backend(exp, real: bool) -> str:
    """Family default: the mock unless --real (or PRISM_BACKEND) says otherwise."""
    from .families import family_for

    env = os.environ.get("PRISM_BACKEND")
    if env:
        return env
    fam = family_for(exp.config)
    return fam.default_backends[0] if real else fam.default_backends[1]


def cmd_status(a):
    from .journal import Journal

    j = Journal(Path(a.dir))
    es = j.entries()
    print(f"{len(es)} iterations, {j.sims_used()} simulations spent")
    b = j.best()
    if b:
        print(f"best: iter {b['iteration']}  FOM {b['fom']:.4g}  {b.get('summary_text','')}  constraints {'ok' if b.get('constraints_ok') else 'violated'}")
    scored = j.scored()
    if scored:
        misses = [e["miss_sigma"] for e in scored if e.get("miss_sigma") is not None]
        if misses:
            within = sum(1 for m in misses if abs(m) <= 2)
            print(f"prediction record: {within}/{len(misses)} within 2 sigma; mean |miss| {sum(abs(m) for m in misses)/len(misses):.2f} sigma")
    print(f"journal: {j.md}")


def cmd_plot(a):
    from .experiment import Experiment
    from .journal import Journal
    from .viz import plot_evolution

    exp = Experiment(a.dir)
    out = plot_evolution(Journal(exp.dir).entries(), exp.dir / (a.out or "evolution.png"), exp.config, log_y=a.log, label_discarded=a.all_labels)
    print("no simulations in the journal yet" if out is None else f"saved {out}")


def cmd_review(a):
    from . import prompts
    from .experiment import Experiment
    from .journal import Journal
    from .llm import get_llm

    exp = Experiment(a.dir)
    j = Journal(exp.dir)
    if not j.md.exists():
        sys.exit("no journal to review")
    llm = get_llm(a.provider, model=a.model) if a.model else get_llm(a.provider)
    user = prompts.REVIEW_USER.format(program=exp.read("program.md"), laws=exp.read("laws.md"), journal_md=j.md.read_text(encoding="utf-8")[-60000:])
    text = llm.complete(prompts.REVIEW_SYSTEM, user, max_tokens=6000)
    out = exp.dir / "review.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nsaved to {out} (laws.md was NOT overwritten - apply the revised version by hand)")


def main(argv=None):
    _load_dotenv()
    p = argparse.ArgumentParser(prog="prism", description="PRISM-Min photonic design agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="create an experiment from a template")
    s.add_argument("dir")
    s.add_argument("--template", default="soi_crossing")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("check", help="build + DRC + sanity + model prediction (no simulation)")
    s.add_argument("dir")
    s.add_argument("--plot", action="store_true")
    s.set_defaults(fn=cmd_check)

    s = sub.add_parser("eval", help="one manual simulation of the current design.py")
    s.add_argument("dir")
    s.add_argument("--backend", default=None, help="mock | mock_pn | lumerical | lumerical_charge_fde (default: family mock, or family real with --real)")
    s.add_argument("--real", action="store_true", help="use the family's real (Lumerical) backend")
    s.add_argument("--manual", action="store_true", required=True, help="acknowledge that the prediction guard is bypassed")
    s.add_argument("--iteration", type=int, default=None)
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("run", help="run the agent loop")
    s.add_argument("dir")
    s.add_argument("--backend", default=None, help="mock | mock_pn | lumerical | lumerical_charge_fde (default: family mock, or family real with --real)")
    s.add_argument("--real", action="store_true", help="use the family's real (Lumerical) backend")
    s.add_argument("--provider", default=None, help="anthropic | openai | scripted (default: PRISM_LLM_PROVIDER)")
    s.add_argument("--model", default=None)
    s.add_argument("--iters", type=int, default=None)
    s.add_argument("--sims", type=int, default=None)
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("status", help="journal summary")
    s.add_argument("dir")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("plot", help="evolution chart: FOM vs experiment, annotated with what the agent changed")
    s.add_argument("dir")
    s.add_argument("--out", default=None, help="file name inside the experiment dir (default evolution.png)")
    s.add_argument("--log", action="store_true", help="log y axis")
    s.add_argument("--all-labels", action="store_true", help="label discarded points too")
    s.set_defaults(fn=cmd_plot)

    s = sub.add_parser("review", help="fresh-session audit of laws.md against the journal")
    s.add_argument("dir")
    s.add_argument("--provider", default=None)
    s.add_argument("--model", default=None)
    s.set_defaults(fn=cmd_review)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
