"""Server-controlled configuration. No secrets are loaded from checked-in files."""
from dataclasses import dataclass
import os
from pathlib import Path

from .providers import JsonHttpProvider, OfflineReferenceProvider, ProviderError

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    fixture_path: Path = ROOT / "fixtures/vocs.json"
    taxonomy_path: Path = ROOT / "fixtures/aspect_catalog.json"
    reference_path: Path = ROOT / "fixtures/reference_context.json"
    output_dir: Path = Path.cwd() / "output/runs"
    api_token: str | None = None
    require_auth: bool = False
    provider_kind: str = "offline"
    provider_endpoint: str = ""
    provider_token: str | None = None
    provider_allowed_hosts: tuple[str, ...] = ()
    allow_external: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        boolean = lambda name: os.getenv(name, "false").lower() == "true"
        result = cls(fixture_path=Path(os.getenv("CX_FIXTURE_PATH", str(ROOT / "fixtures/vocs.json"))),
            taxonomy_path=Path(os.getenv("CX_TAXONOMY_PATH", str(ROOT / "fixtures/aspect_catalog.json"))),
            reference_path=Path(os.getenv("CX_REFERENCE_PATH", str(ROOT / "fixtures/reference_context.json"))),
            output_dir=Path(os.getenv("CX_OUTPUT_DIR", str(Path.cwd() / "output/runs"))),
            api_token=os.getenv("CX_API_TOKEN") or None, require_auth=boolean("CX_REQUIRE_AUTH"),
            provider_kind=os.getenv("CX_PROVIDER", "offline"), provider_endpoint=os.getenv("CX_PROVIDER_ENDPOINT", ""),
            provider_token=os.getenv("CX_PROVIDER_TOKEN") or None,
            provider_allowed_hosts=tuple(h.strip() for h in os.getenv("CX_PROVIDER_ALLOWED_HOSTS", "").split(",") if h.strip()),
            allow_external=boolean("CX_ALLOW_EXTERNAL"))
        if result.require_auth and not result.api_token:
            raise ValueError("CX_API_TOKEN_REQUIRED")
        return result

    def provider(self):
        if self.provider_kind == "offline":
            return OfflineReferenceProvider()
        if self.provider_kind == "http":
            return JsonHttpProvider(self.provider_endpoint, enabled=self.allow_external,
                                   allowed_hosts=set(self.provider_allowed_hosts), token=self.provider_token)
        raise ProviderError("UNKNOWN_PROVIDER_KIND")
