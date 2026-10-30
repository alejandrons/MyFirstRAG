from pathlib import Path
import sys


def main() -> None:
    """Punto de entrada `uv run myfirstrag`: lanza la UI de Streamlit."""
    from streamlit.web import cli as stcli

    app = Path(__file__).parent / "ui" / "app.py"
    sys.argv = ["streamlit", "run", str(app)]
    sys.exit(stcli.main())
