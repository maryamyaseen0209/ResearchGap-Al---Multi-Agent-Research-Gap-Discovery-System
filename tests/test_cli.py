from researchgap.cli import main
from researchgap.core.schemas import BatchResult


def test_cli_writes_usable_json(three_papers, tmp_path):
    paths = []
    for name, content in three_papers:
        path = tmp_path / name
        path.write_bytes(content)
        paths.append(str(path))
    output = tmp_path / "reports" / "processed.json"
    return_code = main([*paths, "--output", str(output)])
    assert return_code == 0
    data = BatchResult.model_validate_json(output.read_text(encoding="utf-8"))
    assert data.ready_for_analysis
