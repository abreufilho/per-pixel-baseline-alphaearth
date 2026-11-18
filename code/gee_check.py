#!/usr/bin/env python3
"""Verifica credenciais e projeto do Google Earth Engine sem baixar artefatos."""
import os
from pathlib import Path

import ee


def main() -> None:
    project = os.environ.get("EE_PROJECT")
    credentials = Path.home() / ".config" / "earthengine" / "credentials"
    if not credentials.exists():
        raise SystemExit(
            f"credenciais Earth Engine ausentes em {credentials}; "
            "rode: docker compose run --rm pipeline earthengine authenticate"
        )
    if not project:
        raise SystemExit("EE_PROJECT ausente; defina no .env ou exporte antes do compose")
    ee.Initialize(project=project)
    value = ee.Number(1).getInfo()
    print(f"✓ Earth Engine OK (project={project}, probe={value})")


if __name__ == "__main__":
    main()
