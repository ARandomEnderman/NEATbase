import logging

from . import config

log = logging.getLogger(__name__)


def coinbase_client():
    from coinbase.rest import RESTClient
    return RESTClient(api_key=config.COINBASE_API_KEY, api_secret=config.COINBASE_API_SECRET)


def resolve_fees():
    """(taker, maker) from your real Coinbase fee tier if a key is loaded, else settings."""
    if config.FEE_SOURCE == "auto" and config.has_api_keys():
        try:
            tier = coinbase_client().get_transaction_summary().to_dict()["fee_tier"]
            taker, maker = float(tier["taker_fee_rate"]), float(tier["maker_fee_rate"])
            log.info("Using your Coinbase fee tier: taker %.3f%%, maker %.3f%%", taker * 100, maker * 100)
            return taker, maker
        except Exception as e:
            log.warning("Could not read fee tier from Coinbase (%s); using TAKER_FEE/MAKER_FEE", e)
    return config.TAKER_FEE, config.MAKER_FEE
