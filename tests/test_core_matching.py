"""Tests for plotverify_core.matching."""
import pytest

from plotverify_core import make_file_entry, match_files


def _img(name):
    return make_file_entry(name, b"\x89PNG")


def _csv(name):
    return make_file_entry(name, b"series,x,y\n")


def test_simple_pair():
    res = match_files([_img("plot1.png")], [_csv("plot1.csv")])
    assert "plot1" in res.pairs
    img, csv = res.pairs["plot1"]
    assert img.filename == "plot1.png"
    assert csv.filename == "plot1.csv"
    assert res.ok_to_proceed


def test_case_insensitive():
    res = match_files([_img("Plot_A.png")], [_csv("plot_a.csv")])
    assert "plot_a" in res.pairs
    # Original filenames are preserved.
    assert res.pairs["plot_a"][0].filename == "Plot_A.png"


def test_extension_independent():
    res = match_files(
        [_img("figure-1.tiff"), _img("figure-2.jpg")],
        [_csv("figure-1.csv"), _csv("figure-2.csv")],
    )
    assert set(res.pairs.keys()) == {"figure-1", "figure-2"}


def test_image_without_csv():
    res = match_files([_img("alone.png")], [])
    assert res.images_without_csv == [_img("alone.png").__class__(
        filename="alone.png", payload=b"\x89PNG", extension=".png",
        canonical_stem="alone",
    )] or res.images_without_csv[0].filename == "alone.png"
    assert res.has_unmatched
    assert not res.ok_to_proceed


def test_csv_without_image():
    res = match_files([], [_csv("orphan.csv")])
    assert res.csvs_without_image[0].filename == "orphan.csv"
    assert not res.pairs


def test_duplicate_image_stem():
    res = match_files(
        [_img("plot.png"), _img("plot.jpg")],
        [_csv("plot.csv")],
    )
    assert "plot" in res.duplicate_image_stems
    # Duplicate stems are NOT paired, even though a CSV exists.
    assert "plot" not in res.pairs
    assert res.has_duplicates
    assert not res.ok_to_proceed


def test_duplicate_csv_stem():
    res = match_files(
        [_img("only.png")],
        [_csv("only.csv"), _csv("only.csv")],  # two CSVs with same content
    )
    assert "only" in res.duplicate_csv_stems


def test_mixed_scenario():
    res = match_files(
        [_img("a.png"), _img("b.png"), _img("dup.png"), _img("dup.jpg")],
        [_csv("a.csv"), _csv("c.csv"), _csv("dup.csv")],
    )
    assert "a" in res.pairs
    assert "b" not in res.pairs  # b has no csv
    # Output is sorted by canonical stem for stable display.
    assert [e.filename for e in res.images_without_csv] == ["b.png"]
    assert sorted(e.filename for e in res.csvs_without_image) == ["c.csv", "dup.csv"]
    assert "dup" in res.duplicate_image_stems
