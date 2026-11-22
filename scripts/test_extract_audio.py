"""Unit tests for scripts/extract_audio.py (stdlib unittest).

Run:  python scripts/test_extract_audio.py
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_audio as ea  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Test vector reads from the already-copied data (not the game install).
LOC1_TXT = PROJECT_ROOT / "data" / "transcripts" / "ru" / "LocationDialogue Location1.txt"


class TestStripTrailingComment(unittest.TestCase):
    def test_strips_trailing_annotation(self):
        self.assertEqual(
            ea.strip_trailing_comment("Мита // is name of Mita (Check for game) [not]"),
            "Мита",
        )

    def test_strips_not_annotation(self):
        self.assertEqual(ea.strip_trailing_comment("... // [not]"), "...")

    def test_no_annotation_unchanged(self):
        self.assertEqual(ea.strip_trailing_comment("Привет, я Мита!"), "Привет, я Мита!")


class TestExtractSpeaker(unittest.TestCase):
    def test_dash_speaker(self):
        self.assertEqual(
            ea.extract_speaker("core console - nope programm - Player"), ("Player", False)
        )

    def test_quest_marker(self):
        self.assertEqual(
            ea.extract_speaker("quest 3 (Я не могу это сделать?) - MitaKind"),
            ("MitaKind", False),
        )

    def test_plain_marker(self):
        self.assertEqual(ea.extract_speaker("Mita"), ("Mita", False))

    def test_last_dash_wins(self):
        self.assertEqual(
            ea.extract_speaker("Last glitch ready - player exit - Mila"), ("Mila", False)
        )

    def test_not_tag_keeps_preceding_speaker(self):
        self.assertEqual(ea.extract_speaker("MitaCreepy - [not]"), ("MitaCreepy", True))
        self.assertEqual(ea.extract_speaker("Mita - [not]"), ("Mita", True))
        self.assertEqual(ea.extract_speaker("Player - [not]"), ("Player", True))

    def test_not_tag_multi_segment(self):
        self.assertEqual(
            ea.extract_speaker("Chibi player - ChibiPlayer - [not]"), ("ChibiPlayer", True)
        )

    def test_not_tag_double_space(self):
        self.assertEqual(
            ea.extract_speaker("Key Search - MitaChibi  - [not]"), ("MitaChibi", True)
        )

    def test_bare_not_tag(self):
        self.assertEqual(ea.extract_speaker("[not]"), (None, True))

    def test_empty(self):
        self.assertEqual(ea.extract_speaker(""), (None, False))


class TestNormalizeCharacter(unittest.TestCase):
    CASES = [
        ("MitaKind", "mita_kind"),
        ("MitaKind ring", "mita_kind"),
        ("MitaKind quest 3 - MitaKind", "mita_kind"),
        ("MitaShorthairs", "mita_shorthairs"),
        ("MitaShorthairs (Показывает куда идти)", "mita_shorthairs"),
        ("MitaCap", "mita_cap"),
        ("Mitacap", "mita_cap"),
        ("MitaCap = Win 2", "mita_cap"),
        ("MitaDream", "mita_dream"),
        ("MitaDream exercise", "mita_dream"),
        ("Mita", "mita"),
        ("MitaBroke", "mita"),
        ("Mitafon", "mita"),
        ("Mitaghost", "mita"),
        ("MitaKiller", "mita"),
        ("Mita killler", "mita"),
        ("Mita card game", "mita"),
        ("Mita (No)", "mita"),
        ("Mita shy", "mita"),
        ("MitaChibi", "mita"),
        ("MitaCreepy", "mita"),
        ("Mila", None),
        ("Player", None),
        ("[not]", None),
        ("core console entry", None),
        ("", None),
        (None, None),
    ]

    def test_buckets(self):
        for speaker, expected in self.CASES:
            with self.subTest(speaker=speaker):
                self.assertEqual(ea.normalize_character(speaker), expected)


class TestCleanText(unittest.TestCase):
    def test_player_placeholder(self):
        self.assertEqual(ea.clean_text("Привет, [player]!"), "Привет, игрок!")

    def test_multiple_placeholders(self):
        self.assertEqual(ea.clean_text("[player] и [player]"), "игрок и игрок")

    def test_no_placeholder(self):
        self.assertEqual(ea.clean_text("Привет"), "Привет")


class TestParseTranscript(unittest.TestCase):
    def test_line10_alignment(self):
        """Test vector: Location1 line 10 == 10.ogg, text 'Привет, я Мита!'."""
        if not LOC1_TXT.exists():
            self.skipTest("data/transcripts/ru not present — run scripts/extract_audio.py first")
        text = LOC1_TXT.read_text(encoding="utf-8-sig")
        line10 = ea.parse_transcript(text)[9]
        self.assertEqual(line10.line_no, 10)
        self.assertEqual(line10.kind, "text")
        self.assertEqual(line10.text, "Привет, я Мита!")
        self.assertIsNone(line10.annotation)

    def test_annotation_split(self):
        lines = ea.parse_transcript("Текст // Player\n")
        self.assertEqual(lines[0].kind, "text")
        self.assertEqual(lines[0].text, "Текст")
        self.assertEqual(lines[0].annotation, "Player")

    def test_blank_lines_counted(self):
        lines = ea.parse_transcript("\n\n// Mita\nПривет\n\n// Player\nПривет, [player]\n")
        self.assertEqual(
            [l.kind for l in lines],
            ["blank", "blank", "marker", "text", "blank", "marker", "text"],
        )
        # Parser keeps [player] intact; clean_text is applied later.
        self.assertEqual(lines[6].text, "Привет, [player]")

    def test_crlf_and_missing_trailing_newline(self):
        lines = ea.parse_transcript("a\r\nb")
        self.assertEqual(len(lines), 2)
        self.assertEqual([l.line_no for l in lines], [1, 2])

    def test_inline_not_is_text(self):
        lines = ea.parse_transcript("Текст // [not]\n")
        self.assertEqual(lines[0].kind, "text")
        self.assertEqual(lines[0].text, "Текст")
        self.assertEqual(lines[0].annotation, "[not]")

    def test_bare_not_marker(self):
        lines = ea.parse_transcript("// [not]\nТекст\n")
        self.assertEqual(lines[0].kind, "marker")
        self.assertEqual(lines[0].marker_body, "[not]")


class TestBuildKnownSpeakers(unittest.TestCase):
    def test_corpus_derived(self):
        known = ea.build_known_speakers(
            {"Player": 289, "Mila": 71, "Mita": 124, "MitaCap": 46,
             "Lamping": 3, "2": 4, "nope version": 2},
            {"Player", "Mila", "Mita", "MitaCap", "Lamping", "2", "nope version"},
        )
        self.assertEqual(known, {"Player", "Mila", "Mita", "MitaCap"})

    def test_mita_prefix_always_known(self):
        known = ea.build_known_speakers({"MitaCap = Win 2": 1}, {"MitaCap = Win 2"})
        self.assertEqual(known, {"MitaCap = Win 2"})

    def test_label_never_inline_not_considered(self):
        # 'Description' is a clean name with high frequency, but it never
        # appears as an inline annotation, so it is not in the set.
        known = ea.build_known_speakers({"Description": 22, "Player": 289}, {"Player"})
        self.assertEqual(known, {"Player"})

    def test_low_frequency_clean_name_excluded(self):
        known = ea.build_known_speakers({"Lamping": 3}, {"Lamping"})
        self.assertEqual(known, set())


class TestResolveLines(unittest.TestCase):
    def setUp(self):
        self.known = ea.build_known_speakers(
            {"Mita": 10, "MitaKind": 10, "Player": 10, "Mila": 10, "Lamping": 3, "2": 4},
            {"Mita", "MitaKind", "Player", "Mila", "Lamping", "2"},
        )

    def _texts(self, transcript: str):
        return [l for l in ea.resolve_lines(ea.parse_transcript(transcript), self.known)
                if l.kind == "text"]

    def test_marker_carry_forward(self):
        texts = self._texts("// Mita\nА\nБ\n")
        self.assertEqual(
            [(t.speaker_raw, t.speaker_source) for t in texts],
            [("Mita", "marker"), ("Mita", "marker")],
        )

    def test_inline_override(self):
        texts = self._texts("// Mita\nА // Player\n")
        self.assertEqual(
            (texts[0].speaker_raw, texts[0].speaker_source, texts[0].inline_note),
            ("Player", "inline", None),
        )

    def test_description_fallback_keeps_marker(self):
        texts = self._texts("// Mita\nА // Points to the board\n")
        self.assertEqual(
            (texts[0].speaker_raw, texts[0].speaker_source, texts[0].inline_note),
            ("Mita", "marker", "Points to the board"),
        )

    def test_unknown_annotation_never_relabels(self):
        # 'Lamping' is not in the known set -> must keep the marker's speaker.
        texts = self._texts("// Mita\nА // Lamping\n")
        self.assertEqual(texts[0].speaker_raw, "Mita")
        self.assertEqual(texts[0].speaker_source, "marker")
        self.assertEqual(texts[0].inline_note, "Lamping")

    def test_not_tag_block(self):
        lines = ea.resolve_lines(
            ea.parse_transcript("// Mita - [not]\nА\n// Player\nБ\n"), self.known
        )
        self.assertEqual(lines[0].speaker_raw, "Mita")
        self.assertTrue(lines[0].unvoiced)
        self.assertTrue(lines[1].unvoiced)  # text line inside [not] block
        self.assertEqual(lines[1].speaker_raw, "Mita")
        self.assertFalse(lines[3].unvoiced)  # next marker clears the flag
        self.assertEqual(lines[3].speaker_raw, "Player")

    def test_speaker_persists_until_next_marker(self):
        lines = ea.resolve_lines(ea.parse_transcript("// Mita\nА\nБ\n// Player\nВ\n"), self.known)
        self.assertEqual(
            [l.speaker_raw for l in lines], ["Mita", "Mita", "Mita", "Player", "Player"]
        )


class TestBuildManifestRowsEndToEnd(unittest.TestCase):
    """Offset rule end-to-end: line N <-> N.ogg, both skip directions.

    Fails if the line->ogg mapping is shifted by one (e.g. get(n + 1)).
    """

    def test_offset_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_out = ea.OUT_AUDIO
            ea.OUT_AUDIO = Path(tmp)
            try:
                cat = "TestCat"
                catdir = Path(tmp) / cat
                catdir.mkdir()
                for n in (1, 2, 4):
                    (catdir / f"{n}.ogg").write_bytes(b"OggS")
                transcript = (
                    "// Mita\n"                    # line 1: marker
                    "Привет // Points to the board\n"  # line 2: text + description
                    "\n"                           # line 3: blank (counted!)
                    "Ты [player] // Player\n"      # line 4: text + inline override
                    "// MitaKind\n"                # line 5: marker
                    "Кольцо!\n"                    # line 6: text, no 6.ogg
                )
                known = ea.build_known_speakers(
                    {"Mita": 10, "Player": 10, "MitaKind": 10},
                    {"Mita", "Player", "MitaKind", "Points to the board"},
                )
                resolved = ea.resolve_lines(ea.parse_transcript(transcript), known)
                oggs = [catdir / f"{n}.ogg" for n in (1, 2, 4)]
                rows, eps_ogg, eps_text, unv = ea.build_manifest_rows(
                    cat, oggs, resolved, {}
                )
                by_id = {r["id"]: r for r in rows}
                # Exact id -> (text, speaker, speaker_source) mapping.
                self.assertEqual(set(by_id), {2, 4})
                self.assertEqual(by_id[2]["text"], "Привет")
                self.assertEqual(
                    (by_id[2]["speaker_raw"], by_id[2]["speaker_source"]),
                    ("Mita", "marker"),
                )
                self.assertEqual(by_id[2]["inline_note"], "Points to the board")
                self.assertEqual(by_id[4]["text"], "Ты игрок")
                self.assertEqual(by_id[4]["text_raw"], "Ты [player]")
                self.assertEqual(
                    (by_id[4]["speaker_raw"], by_id[4]["speaker_source"]),
                    ("Player", "inline"),
                )
                # Both skip directions: 1.ogg has no text (line 1 is a
                # marker); line 6 has text but no 6.ogg.
                self.assertEqual(eps_ogg, 1)
                self.assertEqual(eps_text, 1)
                self.assertEqual(unv, [])
            finally:
                ea.OUT_AUDIO = old_out


if __name__ == "__main__":
    unittest.main(verbosity=2)
