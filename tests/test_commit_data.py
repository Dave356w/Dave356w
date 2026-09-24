"""Invariants of the signed-commit path.

Every ledger commit this repo has made shows as Unverified: a runner has no
signing key, so `git push` cannot produce anything GitHub can verify. The fix
sends the same content through `createCommitOnBranch`, which GitHub signs.
What is pinned here is the part that can silently go wrong -- which files get
sent, what happens when the branch tip moves, and that a failure still lands
the data rather than dropping a slate.
"""

import base64
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import commit_data


def _status(*entries):
    """Join porcelain -z fields the way git emits them."""
    return "".join(f"{e}\0" for e in entries)


def test_written_files_are_additions_and_missing_ones_deletions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "kept.csv").write_text("a,b\n1,2\n")
    adds, dels = commit_data.changed_paths(
        ["data"],
        status_text=_status(" M data/kept.csv", " D data/gone.csv"),
    )
    assert adds == ["data/kept.csv"]
    assert dels == ["data/gone.csv"]


def test_untracked_files_are_sent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "leans_2026-08-27_xw.csv").write_text("x\n")
    adds, _ = commit_data.changed_paths(
        ["data"], status_text=_status("?? data/leans_2026-08-27_xw.csv")
    )
    assert adds == ["data/leans_2026-08-27_xw.csv"]


def test_rename_becomes_an_addition_and_a_deletion(tmp_path, monkeypatch):
    """A rebuild dump is renamed, not rewritten; both halves must be sent."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "rebuild_leans.csv").write_text("x\n")
    adds, dels = commit_data.changed_paths(
        ["data"],
        # git -z emits the new path in the status field and the old path as a
        # bare following field.
        status_text=_status("R  data/rebuild_leans.csv", "data/leans.csv"),
    )
    assert adds == ["data/rebuild_leans.csv"]
    assert dels == ["data/leans.csv"]


def test_contents_are_base64_of_the_file_on_disk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    body = "game_pk,lean\n1,HOU\n"
    (tmp_path / "data" / "led.csv").write_text(body)
    changes = commit_data.file_changes(["data/led.csv"], ["data/old.csv"])
    assert changes["deletions"] == [{"path": "data/old.csv"}]
    sent = changes["additions"][0]
    assert sent["path"] == "data/led.csv"
    assert base64.b64decode(sent["contents"]).decode() == body


def test_no_changes_makes_no_api_call(monkeypatch):
    monkeypatch.setattr(commit_data, "changed_paths", lambda paths: ([], []))
    monkeypatch.setattr(commit_data, "graphql", _explode)
    monkeypatch.setattr(commit_data, "git_commit_and_push", _explode)
    assert commit_data.main(["--branch", "main", "--message", "ledger"]) == 0


def _explode(*args, **kwargs):
    raise AssertionError("should not be called")


class _Moves:
    """remote_head() answers with each tip in turn."""

    def __init__(self, *oids):
        self.oids = list(oids)

    def __call__(self, branch):
        return self.oids.pop(0) if len(self.oids) > 1 else self.oids[0]


def test_retries_when_the_branch_moved_without_touching_our_files(monkeypatch):
    monkeypatch.setattr(commit_data, "remote_head", _Moves("aaa", "bbb", "bbb"))
    monkeypatch.setattr(commit_data, "touches_ours", lambda o, n, ours: [])
    monkeypatch.setattr(commit_data.time, "sleep", lambda s: None)
    seen = []

    def fake_graphql(token, query, variables, timeout=60):
        seen.append(variables["input"]["expectedHeadOid"])
        if len(seen) == 1:
            raise RuntimeError("expected head oid mismatch")
        return {"createCommitOnBranch": {"commit": {"oid": "c" * 40, "url": "u"}}}

    monkeypatch.setattr(commit_data, "graphql", fake_graphql)
    commit = commit_data.api_commit(
        "tok", "o/r", "main", "ledger",
        {"additions": [{"path": "data/x.csv", "contents": ""}], "deletions": []},
    )
    assert commit["oid"] == "c" * 40
    assert seen == ["aaa", "bbb"]


def test_refuses_when_the_intervening_commit_wrote_one_of_our_files(monkeypatch):
    """The `git pull --rebase` conflict refusal, kept in the API path."""
    monkeypatch.setattr(commit_data, "remote_head", _Moves("aaa", "bbb", "bbb"))
    monkeypatch.setattr(
        commit_data, "touches_ours", lambda o, n, ours: ["data/led.csv"]
    )
    monkeypatch.setattr(commit_data.time, "sleep", lambda s: None)

    def fake_graphql(token, query, variables, timeout=60):
        raise RuntimeError("expected head oid mismatch")

    monkeypatch.setattr(commit_data, "graphql", fake_graphql)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        commit_data.api_commit(
            "tok", "o/r", "main", "ledger",
            {"additions": [{"path": "data/led.csv", "contents": ""}],
             "deletions": []},
        )


def test_api_failure_falls_back_to_the_push_that_cannot_lose_a_slate(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "led.csv").write_text("x\n")
    monkeypatch.setattr(
        commit_data, "changed_paths", lambda paths: (["data/led.csv"], [])
    )
    monkeypatch.setattr(commit_data, "api_commit", _explode_api)
    pushed = []
    monkeypatch.setattr(
        commit_data, "git_commit_and_push",
        lambda branch, message: pushed.append((branch, message)),
    )
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    assert commit_data.main(["--branch", "main", "--message", "ledger"]) == 0
    assert pushed == [("main", "ledger")]
    assert "::warning::" in capsys.readouterr().out


def _explode_api(*args, **kwargs):
    raise RuntimeError("graphql exploded")


def test_no_fallback_flag_lets_the_failure_surface(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        commit_data, "changed_paths", lambda paths: (["data/led.csv"], [])
    )
    monkeypatch.setattr(commit_data, "file_changes",
                        lambda adds, dels: {"additions": [], "deletions": []})
    monkeypatch.setattr(commit_data, "api_commit", _explode_api)
    monkeypatch.setattr(commit_data, "git_commit_and_push", _explode)
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    with pytest.raises(RuntimeError, match="graphql exploded"):
        commit_data.main(
            ["--branch", "main", "--message", "ledger", "--no-fallback"]
        )


class TestWorkflowsUseIt:
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _workflow(self, name):
        with open(os.path.join(self.ROOT, ".github", "workflows", name),
                  encoding="utf-8") as fh:
            return fh.read()

    @pytest.mark.parametrize("name,branch", [
        ("build.yml", "--branch main"),
        ("priors-snapshot.yml", "--branch \"${{ github.ref_name }}\""),
        ("lineup-window-collect.yml", "--branch main"),
    ])
    def test_data_is_committed_through_the_signed_path(self, name, branch):
        text = self._workflow(name)
        assert "python commit_data.py" in text
        assert branch in text
        assert "GITHUB_TOKEN: ${{ github.token }}" in text

    @pytest.mark.parametrize("name", ["build.yml", "priors-snapshot.yml",
                                      "lineup-window-collect.yml"])
    def test_no_workflow_pushes_data_directly(self, name):
        """A second, unsigned commit path would quietly undo this."""
        lines = [ln.strip() for ln in self._workflow(name).splitlines()
                 if not ln.strip().startswith("#")]
        for forbidden in ("git add data/", "git push"):
            assert not [ln for ln in lines if ln.startswith(forbidden)], (
                f"{name} still commits data/ with git; that commit lands "
                "unverified and bypasses commit_data.py"
            )

    @pytest.mark.parametrize("name", ["build.yml", "priors-snapshot.yml",
                                      "lineup-window-collect.yml"])
    def test_validation_still_precedes_the_commit(self, name):
        text = self._workflow(name)
        assert text.index("python validate_data_files.py") < text.index(
            "python commit_data.py"
        )


class TestLineupReportPublish:
    """The collect workflow may write exactly one file, from one job, on
    main-branch family runs -- and must not queue in `site-build`, where a
    pending run of its own could replace a pending build.

    Text checks rather than a YAML parse: PyYAML is not in requirements.txt,
    and the test job installs only that plus pytest.
    """
    ROOT = TestWorkflowsUseIt.ROOT

    def _parts(self):
        with open(os.path.join(self.ROOT, ".github", "workflows",
                               "lineup-window-collect.yml"),
                  encoding="utf-8") as fh:
            text = fh.read()
        code = "\n".join(ln for ln in text.splitlines()
                         if not ln.strip().startswith("#"))
        head, jobs = code.split("\njobs:\n", 1)
        collect, publish = jobs.split("\n  publish:\n", 1)
        return head, collect, publish

    def test_collect_job_stays_read_only(self):
        head, collect, _ = self._parts()
        assert "permissions:\n  contents: read" in head
        assert "permissions:" not in collect
        assert "commit_data.py" not in collect

    def test_publish_commits_only_the_report(self):
        _, _, publish = self._parts()
        assert "contents: write" in publish
        assert "--path data/pitcher_lineup_report.txt" in publish
        assert publish.count("commit_data.py") == 1

    def test_publish_skips_pr_and_single_slate_runs(self):
        _, _, publish = self._parts()
        assert "needs: collect" in publish
        assert "github.event_name != 'pull_request'" in publish
        assert "refs/heads/main" in publish
        assert "!inputs.date" in publish

    def test_not_in_the_build_concurrency_group(self):
        head, collect, publish = self._parts()
        for part in (head, collect, publish):
            assert "concurrency:" not in part
            assert "site-build" not in part


class TestVelocityMigrationIsNeverStale:
    """The migration rewrites the whole ledger file, and it queues behind
    site-build. It must work from the tip that build left, and refuse to
    commit if main moved -- its first run did neither and undid a build."""

    def _text(self):
        with open(os.path.join(TestWorkflowsUseIt.ROOT, ".github", "workflows",
                               "velocity-migration.yml"), encoding="utf-8") as fh:
            return fh.read()

    def test_checks_out_the_latest_main(self):
        t = self._text()
        assert "uses: actions/checkout@v4\n        with:\n          ref: main" in t

    def test_refuses_when_main_moved_before_compute_and_commit(self):
        t = self._text()
        guard = 'test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"'
        assert t.count(guard) == 2
        assert t.index(guard) < t.index("python reconstruct_v14_velocity.py")
        assert t.rindex(guard) < t.index("python commit_data.py")

    def test_serialised_with_the_build(self):
        t = self._text()
        assert "group: site-build" in t and "cancel-in-progress: false" in t
