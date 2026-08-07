"""Selection-reducer logic for overlay multi-select."""
from shiny_app.app import _sel_anchor, _sel_part, _sel_pids, _update_selection


def test_plain_click_replaces():
    sel = {"pids": ["A#0", "A#1"], "anchor": "A#1", "part": "center"}
    new = _update_selection(sel, "B#0", "upper", shift=False)
    assert new == {"pids": ["B#0"], "anchor": "B#0", "part": "upper"}


def test_plain_click_from_empty():
    new = _update_selection(None, "A#0", "center", shift=False)
    assert new == {"pids": ["A#0"], "anchor": "A#0", "part": "center"}


def test_shift_click_adds_and_moves_anchor():
    sel = _update_selection(None, "A#0", "center", shift=False)
    sel = _update_selection(sel, "A#1", "center", shift=True)
    assert _sel_pids(sel) == ["A#0", "A#1"]
    assert _sel_anchor(sel) == "A#1"
    assert _sel_part(sel) == "center"


def test_shift_click_removes_member():
    sel = {"pids": ["A#0", "A#1", "A#2"], "anchor": "A#2", "part": "center"}
    new = _update_selection(sel, "A#1", "center", shift=True)
    assert _sel_pids(new) == ["A#0", "A#2"]
    assert _sel_anchor(new) == "A#2"


def test_shift_click_removing_anchor_falls_back_to_last():
    sel = {"pids": ["A#0", "A#1", "A#2"], "anchor": "A#2", "part": "center"}
    new = _update_selection(sel, "A#2", "center", shift=True)
    assert _sel_pids(new) == ["A#0", "A#1"]
    assert _sel_anchor(new) == "A#1"


def test_shift_click_removing_last_clears_selection():
    sel = {"pids": ["A#0"], "anchor": "A#0", "part": "upper"}
    assert _update_selection(sel, "A#0", "center", shift=True) is None


def test_part_forced_center_for_multi():
    sel = _update_selection(None, "A#0", "upper", shift=False)
    assert _sel_part(sel) == "upper"
    sel = _update_selection(sel, "A#1", "lower", shift=True)
    assert _sel_part(sel) == "center"
    # dropping back to one point restores the clicked part
    sel = _update_selection(sel, "A#1", "lower", shift=True)
    assert _sel_pids(sel) == ["A#0"]
    assert _sel_part(sel) == "lower"


def test_sel_helpers_on_none():
    assert _sel_pids(None) == []
    assert _sel_anchor(None) is None
    assert _sel_part(None) == "center"
