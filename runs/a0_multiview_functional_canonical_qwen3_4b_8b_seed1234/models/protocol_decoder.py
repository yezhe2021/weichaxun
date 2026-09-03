from .writer import ProtocolTransform


class ProtocolDecoder(ProtocolTransform):
    """Source-blind temporary decoder; it receives no sender metadata."""

