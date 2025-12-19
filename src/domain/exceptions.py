"""
Domain exceptions for the Apex system.

Implements a hierarchy distinguishing between recoverable runtime errors
(network glitches, temporary data unavailability) and fatal errors
(configuration issues, programming bugs) that require system shutdown.
"""

class ApexError(Exception):
    """Base class for all Apex domain exceptions."""
    pass


class RecoverableError(ApexError):
    """
    Errors that the system can recover from without restarting.
    
    Examples:
    - Temporary network disconnection
    - Stale market data
    - Broker API temporary unavailability
    - Recoverable state inconsistencies
    """
    pass


class FatalError(ApexError):
    """
    Critical errors requiring system shutdown or operator intervention.
    
    Examples:
    - Invalid configuration
    - Missing required components
    - Unrecoverable data corruption
    - Broker authentication failure (wrong credentials)
    """
    pass


class MarketDataError(RecoverableError):
    """Issues fetching or processing market data."""
    pass


class ExecutionError(RecoverableError):
    """Order submission or management failures."""
    pass


class RiskCheckError(RecoverableError):
    """Risk validation failures (logic valid, but trade rejected)."""
    pass


class ConfigurationError(FatalError):
    """Invalid system configuration."""
    pass


class InitializationError(FatalError):
    """Component failed to initialize correctly."""
    pass
