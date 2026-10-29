import logging

from . import config

log = logging.getLogger(__name__)


def coinbase_client():
    from coinbase.rest import RESTClient
    return RESTClient(api_key=config.COINBASE_API_KEY, api_secret=config.COINBASE_API_SECRET)


def resolve_taker_fee():
    """Your real taker fee tier if API keys are configured, otherwise TAKER_FEE."""
    if config.FEE_SOURCE == "auto" and config.has_api_keys():
        try:
            summary = coinbase_client().get_transaction_summary().to_dict()
            rate = float(summary["fee_tier"]["taker_fee_rate"])
            log.info("Using taker fee %.4f%% from your Coinbase fee tier", rate * 100)
            return rate
        except Exception as e:
            log.warning("Could not read fee tier from Coinbase (%s); using TAKER_FEE", e)
    return config.TAKER_FEE
