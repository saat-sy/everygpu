"""The partition, verify and upload workflow. Needs the heavy dependencies."""

import collections
import json
from pathlib import Path

from model_partitioner.artifact import hub, loader
from model_partitioner.partition import adapters, partitioner, source
from model_partitioner.utils import constants, models, path_utils, progress
from model_partitioner.verification import verifier
from model_partitioner.workflow import report as report_module


class Workflow:
    """One run: owns the config, pinned source facts, results and the report."""

    def __init__(
        self,
        cfg: models.RunConfig,
        run: report_module.Report,
        tracker: progress.Progress | None = None,
    ):
        self.cfg = cfg
        self.report = run
        self.progress = tracker or run.progress
        self.adapter = adapters.adapter_for_repo(cfg.model)
        self.revision: str | None = None
        self.hub_files: dict[str, models.FileInfo] = {}
        self.source_root: Path | None = cfg.source_dir
        self.config: dict = {}
        self.weight_bytes = 0
        self.results: list[models.CheckResult] = []

    def run(self) -> int:
        """Run the remaining stages in order; stop at the first one that does not pass."""
        steps = {
            constants.STAGE_ACCESS: self.check_access,
            constants.STAGE_FETCH: self.fetch_source,
            constants.STAGE_VALIDATE: self.validate_source,
            constants.STAGE_PARTITION: self.partition,
            constants.STAGE_VERIFY: self.verify,
            constants.STAGE_UPLOAD: self.upload,
        }
        try:
            for name, step in steps.items():
                if name in self.report.names:
                    with self.report.stage(name):
                        step()
        except BaseException as error:
            return self.report.finish(report_module.outcome_of(error), error)
        return self.report.finish(constants.PASSED)

    def _check(self, name: str, detail: str = "") -> None:
        self.report.record_check(name, constants.PASSED, detail)

    def _record_identity(self) -> None:
        self.report.data["identity"] = {
            "model": self.cfg.model,
            "family": self.adapter.family,
            "source": "local" if self.cfg.source_dir else "huggingface",
            "source_dir": str(self.cfg.source_dir) if self.cfg.source_dir else None,
            "revision": self.revision,
        }

    def _record_manifest(self) -> models.Manifest:
        manifest = loader.read_manifest(self.cfg.artifact_dir)
        files = loader.declared_files(manifest)
        self.report.data["manifest"] = {
            "revision": manifest.revision,
            "units": len(manifest.units),
            "tensors": len(manifest.tensors),
            "partition_files": len(manifest.files),
            "asset_files": len(manifest.assets),
            "declared_bytes": sum(info.size for info in files.values()),
        }
        return manifest

    def check_access(self) -> None:
        cfg = self.cfg
        if cfg.upload:
            self._check("upload credentials", hub.check_upload_access(cfg.upload))
        if cfg.verify_only:
            manifest = loader.read_manifest(cfg.verify_only)
            self.revision, self.config = manifest.revision, manifest.config
            self.weight_bytes = sum(t.nbytes for t in manifest.source_tensors.values())
        if cfg.source_dir:
            self.config = source.LocalSource(cfg.source_dir).config()
            self.adapter.check_config(self.config)
            self.weight_bytes = sum(
                (cfg.source_dir / name).stat().st_size
                for name in source.LocalSource(cfg.source_dir).files
            )
            self._check("local source", f"{cfg.source_dir}, config matches {cfg.model}")
        if not cfg.source_dir:
            self._check_hub_source()
        if cfg.verify == constants.VERIFY_GPU:
            self._check_gpu()
        self._check_disk(download=True, partition=True, scratch=True)
        self._record_identity()

    def _check_hub_source(self) -> None:
        cfg = self.cfg
        if not cfg.verify_only:
            self.revision = hub.resolve_revision(cfg.model)
        config_path = hub.fetch_config(cfg.model, self.revision)
        self.config = json.loads(Path(config_path).read_text())
        self.adapter.check_config(self.config)
        self.hub_files = hub.remote_files(cfg.model, self.revision)
        if not any(name.endswith(".safetensors") for name in self.hub_files):
            raise models.StageError(
                f"{cfg.model}@{self.revision} declares no safetensors files with Hub sha256"
            )
        if not cfg.verify_only:
            self.weight_bytes = sum(
                i.size for n, i in self.hub_files.items() if n.endswith(".safetensors")
            )
        self._check("source access", f"{cfg.model} @ {self.revision}")

    def _check_gpu(self) -> None:
        from model_partitioner.verification import output_check

        blocker = output_check.gpu_blocker() or self.adapter.output_blocker(
            self.config, True
        )
        if blocker:
            raise models.StageError(blocker)
        self._check("gpu", "CUDA devices usable by this PyTorch build")

    def _check_disk(self, download=False, partition=False, scratch=False) -> None:
        """Sum needs per filesystem, then compare each with its free space."""
        needs = []
        if download and not self.cfg.source_dir:
            missing = hub.missing_download_bytes(
                self.cfg.model, self.revision, self.hub_files
            )
            needs.append(("download", Path(hub.cache_dir()), missing))
        if partition and not self.cfg.verify_only:
            needs.append(("partitions", self.cfg.output, self.weight_bytes))
        if scratch and self.cfg.verify:
            needs.append(
                (
                    "output-check scratch",
                    self.cfg.artifact_dir.resolve().parent,
                    self.weight_bytes,
                )
            )
        totals = collections.defaultdict(lambda: [0, [], None])
        for label, path, size in needs:
            entry = totals[path_utils.device_of(path)]
            entry[0] += size
            entry[1].append(f"{label} {size:,}")
            entry[2] = path
        for need, labels, path in totals.values():
            free = path_utils.free_bytes(path)
            required = need + constants.DISK_MARGIN_BYTES
            detail = f"{path}: {', '.join(labels)}; need {required:,} incl. margin, {free:,} free"
            if free < required:
                raise models.StageError(f"not enough disk space: {detail}")
            self._check("disk space", detail)

    def fetch_source(self) -> None:
        self.progress.say(
            "downloading (skipped files already cached); Hugging Face progress below"
        )
        path = hub.fetch_source(self.cfg.model, self.revision)
        if not self.cfg.verify_only:
            self.source_root = path

    def validate_source(self) -> None:
        with source.LocalSource(self.source_root) as src:
            self.adapter.check_config(src.config())
            with self.progress.task("read tensor headers and plan units"):
                units = partitioner.plan_source(src, self.adapter)
            self.progress.say(partitioner.format_summary(units))
        partitioner.check_destination(self.source_root, self.cfg.output)
        self._check("source weights", "layout, shards, index and dtypes supported")

    def partition(self) -> None:
        self._check_disk(partition=True, scratch=True)
        hub_source = self.cfg.source_dir is None
        hashes = {n: i.sha256 for n, i in self.hub_files.items()}
        with source.LocalSource(self.source_root) as src:
            partitioner.partition(
                src,
                self.adapter,
                self.cfg.output,
                self.cfg.model,
                self.revision if hub_source else None,
                hashes if hub_source else None,
                tracker=self.progress,
            )
        manifest = self._record_manifest()
        self.revision = manifest.revision
        self._record_identity()
        self.progress.say(
            f"wrote {self.report.data['manifest']['partition_files']} files to {self.cfg.output}"
        )

    def verify(self) -> None:
        cfg = self.cfg
        self._check_disk(scratch=True)
        if cfg.verify_only:
            self._record_manifest()
        results = verifier.verify_structure(
            cfg.artifact_dir, cfg.source_dir, self.progress
        )
        if cfg.verify:
            results.append(self._compare_outputs(results))
        self.results = results
        for result in results:
            self.report.record_check(result.name, result.status, result.detail)
        self._gate()

    def _compare_outputs(self, results: list[models.CheckResult]) -> models.CheckResult:
        if any(r.status == constants.FAILED for r in results):
            return models.CheckResult(
                constants.OUTPUT_CHECK_NAME,
                constants.UNVERIFIED,
                "skipped: structural checks failed",
            )
        from model_partitioner.verification import output_check

        options = (
            models.RunOptions("cuda", "auto")
            if self.cfg.verify == constants.VERIFY_GPU
            else models.RunOptions()
        )
        return output_check.check_outputs(
            self.cfg.artifact_dir, options, self.cfg.source_dir, self.progress
        )

    def _unmet(self) -> list[models.CheckResult]:
        """Required checks that did not PASS, including ones that never ran."""
        by_name = {r.name: r for r in self.results}
        required = verifier.required_checks(bool(self.cfg.verify))
        return [
            by_name.get(name, models.CheckResult(name, constants.UNVERIFIED, "not run"))
            for name in required
            if by_name.get(name) is None or by_name[name].status != constants.PASSED
        ] + [
            r
            for r in self.results
            if r.status != constants.PASSED and r.name not in required
        ]

    def _gate(self) -> None:
        unmet = self._unmet()
        if not unmet:
            return
        failed = any(r.status == constants.FAILED for r in unmet)
        names = ", ".join(f"{r.name} {r.status}" for r in unmet)
        if self.cfg.upload:
            reason = f"verification not fully passed ({names})"
            self.progress.say(f"upload blocked: {reason}")
            self.report.data["upload"] = {"status": "BLOCKED", "reason": reason}
        raise models.StageError(
            f"verification not passed: {names}",
            constants.FAILED if failed else constants.INCOMPLETE,
        )

    def upload(self) -> None:
        cfg = self.cfg
        if self._unmet() or not self.results:
            raise models.StageError("upload refused: verification has not fully passed")
        self.report.data["upload"] = {"destination": cfg.upload, "status": "STARTED"}
        self._check("upload credentials (recheck)", hub.check_upload_access(cfg.upload))
        created = hub.ensure_repo(cfg.upload)
        self.progress.say(
            f"{'created private repo' if created else 'existing repo, visibility unchanged'}: {cfg.upload}"
        )
        revision = hub.upload_artifact(cfg.artifact_dir, cfg.upload)
        upload = self.report.data["upload"]
        upload.update(
            created=created, revision=revision, status="UPLOADED", verified=False
        )
        with self.progress.task("remote integrity check against manifest"):
            problems = hub.check_upload(cfg.artifact_dir, cfg.upload, revision)
        if problems:
            upload.update(remote_integrity=constants.FAILED, problems=problems[:20])
            self.report.record_check(
                "remote integrity",
                constants.FAILED,
                f"{len(problems)} problems, first: {problems[:3]}",
            )
            raise models.StageError(
                "uploaded files do not match the manifest; remote changes are not rolled back"
            )
        upload.update(
            remote_integrity=constants.PASSED,
            verified=True,
            url=f"{constants.HUB_URL}{cfg.upload}",
        )
        self.report.record_check(
            "remote integrity", constants.PASSED, f"revision {revision}"
        )
