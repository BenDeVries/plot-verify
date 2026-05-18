"""Verifies plotverify_core has zero Streamlit dependency.

Runs the import in a subprocess so the sys.meta_path mutation does not leak
into other test files.
"""
import subprocess
import sys
import textwrap


def test_imports_without_streamlit():
    src = textwrap.dedent("""
        import sys

        class _Blocker:
            def find_spec(self, name, *a, **k):
                if name == "streamlit" or name.startswith("streamlit."):
                    raise ImportError("simulated missing streamlit")
                return None
        sys.meta_path.insert(0, _Blocker())

        import plotverify_core as pkg
        assert callable(pkg.is_valid_hex)
        assert callable(pkg.compute_calibration)
        assert callable(pkg.delta_e_mask)
        assert callable(pkg.load_csv)
        assert callable(pkg.decode_image_bytes)
        assert callable(pkg.init_series_states)
        print("OK")
    """)
    res = subprocess.run(
        [sys.executable, "-c", src],
        capture_output=True, text=True,
        cwd=__file__.rsplit("/", 2)[0],
    )
    assert res.returncode == 0, (
        f"plotverify_core failed to import without streamlit:\n"
        f"STDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    )
    assert "OK" in res.stdout
