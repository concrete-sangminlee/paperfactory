"""End-to-end tests for the command-line interface (cli.py)."""

import json
import os

import pytest

import cli

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "examples", "sample_paper.json")


def run(capsys, *argv):
    code = cli.main([str(a) for a in argv])
    out, err = capsys.readouterr()
    return code, out, err


@pytest.fixture
def bad_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    return str(path)


class TestBasics:
    def test_no_command_prints_help(self, capsys):
        code, out, _ = run(capsys)
        assert code == 0
        assert "paperfactory" in out

    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["--version"])
        assert exc.value.code == 0
        assert cli.__version__ in capsys.readouterr().out

    def test_all_commands_have_help(self, capsys):
        parser = cli.build_parser()
        sub = next(a for a in parser._actions if a.dest == "command")
        assert len(sub.choices) >= 18
        for name in sub.choices:
            with pytest.raises(SystemExit) as exc:
                cli.main([name, "--help"])
            assert exc.value.code == 0
        capsys.readouterr()


class TestJournals:
    def test_text(self, capsys):
        code, out, _ = run(capsys, "journals")
        assert code == 0
        assert "jweia" in out and "Journal of Constructional Steel Research" in out
        assert len(out.strip().splitlines()) == 15

    def test_json(self, capsys):
        code, out, _ = run(capsys, "journals", "--json")
        assert code == 0
        assert len(json.loads(out)) == 15


class TestPaperCommands:
    def test_check(self, capsys):
        code, out, _ = run(capsys, "check", "-p", SAMPLE, "-j", "jweia")
        assert code == 0
        assert "Quality Score" in out

    def test_check_strict_fails_short_paper(self, capsys):
        code, _, _ = run(capsys, "check", "-p", SAMPLE, "-j", "jweia", "--strict")
        assert code == 1

    def test_check_json(self, capsys):
        code, out, _ = run(capsys, "check", "-p", SAMPLE, "-j", "JWEIA", "--json")
        data = json.loads(out)
        assert code == 0 and "score" in data and "checks" in data

    def test_unknown_journal_is_clean_error(self, capsys):
        code, out, err = run(capsys, "check", "-p", SAMPLE, "-j", "jweai")
        assert code == 1
        assert "Did you mean: jweia" in err
        assert "Traceback" not in err

    def test_missing_file(self, capsys):
        code, _, err = run(capsys, "check", "-p", "nope.json", "-j", "jweia")
        assert code == 1 and "File not found" in err

    def test_invalid_json(self, capsys, bad_json):
        code, _, err = run(capsys, "review", "-p", bad_json, "-j", "jweia")
        assert code == 1 and "Invalid JSON" in err

    def test_non_object_json(self, capsys, tmp_path):
        path = tmp_path / "list.json"
        path.write_text("[1, 2]", encoding="utf-8")
        code, _, err = run(capsys, "analytics", "-p", str(path))
        assert code == 1 and "JSON object" in err

    def test_review(self, capsys):
        code, out, _ = run(capsys, "review", "-p", SAMPLE, "-j", "eng_structures")
        assert code == 0 and out.strip()

    def test_review_forwards_figures(self, capsys, tmp_path):
        # The sample paper lists 6 figure captions; passing only 2 figure files must be
        # reflected in the review, proving --figures reaches review_paper.
        figs = []
        for i in range(2):
            f = tmp_path / f"fig_{i}.png"
            f.write_bytes(b"\x89PNG\r\n\x1a\n")
            figs.append(str(f))
        code, out, _ = run(
            capsys, "review", "-p", SAMPLE, "-j", "jweia", "-f", ",".join(figs), "--json"
        )
        comments = [c["comment"] for c in json.loads(out)["comments"]]
        assert code == 0
        assert any("Only 2 figures" in c for c in comments)
        _, out2, _ = run(capsys, "review", "-p", SAMPLE, "-j", "jweia", "--json")
        assert not any("Only 2 figures" in c["comment"] for c in json.loads(out2)["comments"])

    def test_review_json(self, capsys):
        code, out, _ = run(capsys, "review", "-p", SAMPLE, "-j", "jweia", "--json")
        assert code == 0 and "decision" in json.loads(out)

    def test_checklist(self, capsys):
        code, out, _ = run(capsys, "checklist", "-p", SAMPLE, "-j", "jweia")
        assert code == 0 and "Ready to submit" in out

    def test_checklist_json(self, capsys):
        code, out, _ = run(capsys, "checklist", "-p", SAMPLE, "-j", "jweia", "--json")
        assert code == 0 and "ready" in json.loads(out)

    def test_cover_letter_to_file(self, capsys, tmp_path):
        out_file = tmp_path / "sub" / "letter.txt"
        code, out, _ = run(
            capsys, "cover-letter", "-p", SAMPLE, "-j", "jweia", "-e", "Prof. Kim", "-o", out_file
        )
        assert code == 0
        assert "Prof. Kim" in out_file.read_text(encoding="utf-8")

    def test_analytics_verbose(self, capsys):
        code, out, _ = run(capsys, "analytics", "-p", SAMPLE, "-v")
        assert code == 0 and "Section balance" in out

    def test_analytics_json(self, capsys):
        code, out, _ = run(capsys, "analytics", "-p", SAMPLE, "--json")
        assert code == 0 and "readability" in json.loads(out)

    def test_abstract_improve(self, capsys):
        code, out, _ = run(capsys, "abstract", "-p", SAMPLE)
        assert code == 0 and "Abstract score" in out

    def test_abstract_generate(self, capsys):
        code, out, _ = run(capsys, "abstract", "-p", SAMPLE, "-g", "-m", "120")
        assert code == 0 and "Generated abstract" in out

    def test_abstract_missing(self, capsys, tmp_path):
        path = tmp_path / "p.json"
        path.write_text(json.dumps({"title": "T", "sections": []}), encoding="utf-8")
        code, _, err = run(capsys, "abstract", "-p", str(path))
        assert code == 1 and "No abstract" in err

    def test_convert_refs(self, capsys):
        code, out, _ = run(capsys, "convert-refs", "-p", SAMPLE, "-s", "apa")
        assert code == 0 and out.strip()

    def test_validate_refs(self, capsys):
        code, out, _ = run(capsys, "validate-refs", "-p", SAMPLE, "-j", "jweia", "--json")
        data = json.loads(out)
        assert code == 0 and data["count"] > 0

    def test_validate_refs_strict_detects_duplicates(self, capsys, tmp_path):
        ref = "[1] A. Kim, Wind loads, J. Wind Eng. 1 (2023) 1-2. https://doi.org/10.1/x"
        path = tmp_path / "p.json"
        path.write_text(json.dumps({"references": [ref, ref]}), encoding="utf-8")
        code, out, _ = run(capsys, "validate-refs", "-p", str(path), "-j", "jweia", "--strict")
        assert code == 1 and "duplicate" in out

    def test_reformat(self, capsys, tmp_path):
        out_file = tmp_path / "re.json"
        code, _, _ = run(
            capsys, "reformat", "-p", SAMPLE, "--from", "jweia", "--to", "asce_jse", "-o", out_file
        )
        assert code == 0
        assert json.loads(out_file.read_text(encoding="utf-8"))["title"]


class TestExport:
    @pytest.mark.parametrize(
        "fmt,suffixes", [("docx", [".docx"]), ("latex", [".tex", ".bib"]), ("pdf", [".pdf"])]
    )
    def test_each_format(self, capsys, tmp_path, fmt, suffixes):
        code, out, _ = run(
            capsys, "export", "-p", SAMPLE, "-j", "jweia", "--format", fmt, "-o", str(tmp_path)
        )
        assert code == 0, out
        produced = {os.path.splitext(f)[1] for f in os.listdir(tmp_path)}
        for s in suffixes:
            assert s in produced

    def test_json_and_multiple_formats(self, capsys, tmp_path):
        code, out, _ = run(
            capsys,
            "export",
            "-p",
            SAMPLE,
            "-j",
            "eng_structures",
            "--format",
            "docx",
            "pdf",
            "docx",
            "-o",
            str(tmp_path),
            "--json",
        )
        data = json.loads(out)
        assert code == 0 and set(data) == {"docx", "pdf"}
        assert all(os.path.exists(p) for p in data.values())

    def test_missing_figure_warns(self, capsys, tmp_path):
        code, _, err = run(
            capsys, "export", "-p", SAMPLE, "-j", "jweia", "-f", "nope.png", "-o", str(tmp_path)
        )
        assert code == 0 and "not found" in err


class TestBib:
    def test_import_bib(self, capsys, tmp_path):
        bib = tmp_path / "r.bib"
        bib.write_text(
            "@article{kim2023,\n  author={Kim, A. and Lee, B.},\n  title={Wind loads},\n"
            "  journal={J. Wind Eng.},\n  year={2023},\n  volume={1},\n  pages={1--2}\n}\n",
            encoding="utf-8",
        )
        out_json = tmp_path / "refs.json"
        code, out, _ = run(capsys, "import-bib", "-b", str(bib), "-o", str(out_json))
        assert code == 0 and "Imported 1" in out
        assert len(json.loads(out_json.read_text(encoding="utf-8"))["references"]) == 1

    def test_import_bib_missing(self, capsys):
        code, _, err = run(capsys, "import-bib", "-b", "missing.bib")
        assert code == 1 and "File not found" in err


class TestTemplates:
    def test_list(self, capsys):
        code, out, _ = run(capsys, "templates")
        assert code == 0 and "ml_comparison" in out

    def test_generate(self, capsys, tmp_path):
        out_file = tmp_path / "s.json"
        code, _, _ = run(
            capsys, "templates", "-g", "ml_comparison", "-t", "Wind", "-j", "jweia", "-o", out_file
        )
        assert code == 0
        assert "sections" in json.loads(out_file.read_text(encoding="utf-8"))

    def test_unknown_template(self, capsys):
        code, _, err = run(capsys, "templates", "-g", "nope")
        assert code == 1 and "Unknown template" in err


class TestSources:
    def test_sources(self, capsys):
        code, out, _ = run(capsys, "sources", "-t", "wind pressure on low-rise buildings")
        assert code == 0 and "TPU" in out

    def test_sources_json(self, capsys):
        code, out, _ = run(capsys, "sources", "-t", "earthquake ground motion", "--json")
        assert code == 0 and isinstance(json.loads(out), list)


class TestPipelineCommands:
    def test_lifecycle(self, capsys):
        code, out, _ = run(capsys, "new", "-t", "Wind pressure ML", "-j", "jweia")
        assert code == 0 and "Pipeline created" in out
        run_id = out.split("Pipeline created: ")[1].split()[0]

        code, out, _ = run(capsys, "status")
        assert code == 0 and run_id in out
        code, out, _ = run(capsys, "status", "-r", run_id, "--json")
        assert json.loads(out)["run_id"] == run_id
        code, out, _ = run(capsys, "resume", "-r", run_id)
        assert code == 0 and "literature" in out.lower()
        code, out, _ = run(capsys, "runs")
        assert code == 0 and run_id in out
        code, out, _ = run(capsys, "runs", "--json")
        assert json.loads(out)[0]["run_id"] == run_id

    def test_status_without_runs(self, capsys):
        code, _, err = run(capsys, "status")
        assert code == 1 and "No pipeline found" in err

    def test_resume_unknown(self, capsys):
        code, _, err = run(capsys, "resume", "-r", "run_missing")
        assert code == 1 and "not found" in err

    def test_new_invalid_journal(self, capsys):
        code, _, err = run(capsys, "new", "-t", "Topic", "-j", "bogus")
        assert code == 1 and "Guideline not found" in err

    def test_runs_empty(self, capsys):
        code, out, _ = run(capsys, "runs")
        assert code == 0 and "No pipeline runs" in out
