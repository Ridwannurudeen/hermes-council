"""CLI for hermes-council: skill installation."""

import argparse
import shutil
import sys
from pathlib import Path


def _get_skills_source() -> Path:
    """Locate bundled skills directory."""
    package_dir = Path(__file__).resolve().parent
    # Skills are at repo_root/skills/council/ relative to src/hermes_council/
    skills_dir = package_dir.parent.parent / "skills" / "council"
    return skills_dir


def install_skills(force: bool = False):
    """Copy council skills to ~/.hermes/skills/council/."""
    source = _get_skills_source()
    if not source.exists():
        print(f"Error: Skills source not found at {source}", file=sys.stderr)
        sys.exit(1)

    target = Path.home() / ".hermes" / "skills" / "council"

    if target.exists() and not force:
        print(
            f"Skills already installed at {target}\n"
            f"Use --force to overwrite.",
            file=sys.stderr,
        )
        sys.exit(1)

    if target.exists():
        shutil.rmtree(target)

    shutil.copytree(source, target)
    print(f"Installed council skills to {target}")

    for path in sorted(target.rglob("*.md")):
        print(f"  {path.relative_to(target)}")


def main():
    """Entry point for hermes-council CLI."""
    parser = argparse.ArgumentParser(
        prog="hermes-council",
        description="hermes-council: Adversarial deliberation tools",
    )
    subparsers = parser.add_subparsers(dest="command")

    install_parser = subparsers.add_parser(
        "install-skills", help="Install council skills to ~/.hermes/skills/"
    )
    install_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing skills"
    )

    args = parser.parse_args()

    if args.command == "install-skills":
        install_skills(force=args.force)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
