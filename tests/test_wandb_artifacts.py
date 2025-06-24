from pathlib import Path


class DummyArtifact:
    def __init__(self, path: Path):
        self.path = Path(path)

    def download(self) -> str:
        return str(self.path)


def test_model_artifact_download(tmp_path):
    art_dir = tmp_path / "artifact"
    art_dir.mkdir()
    (art_dir / "model.pkl").write_text("model")
    (art_dir / "preprocessing_pipeline.pkl").write_text("pipeline")

    model_artifact = DummyArtifact(art_dir)
    download_dir = Path(model_artifact.download())
    assert (download_dir / "model.pkl").is_file()
    assert (download_dir / "preprocessing_pipeline.pkl").is_file()
