class BenchmarkV2Error(RuntimeError):
    """Base error for the independent benchmark runtime."""


class ProtocolError(BenchmarkV2Error):
    pass


class ModelUnavailableError(BenchmarkV2Error):
    pass


class ContractError(BenchmarkV2Error, ValueError):
    pass


class ArtifactError(BenchmarkV2Error):
    pass

