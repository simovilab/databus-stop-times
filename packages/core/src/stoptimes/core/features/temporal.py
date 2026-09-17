from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

try:
    import zoneinfo  # py3.9+
except ImportError:  # pragma: no cover
    from backports import zoneinfo  # type: ignore

def _get_holiday_calendar(region: str | None):
    """
    Build a holiday calendar for ``COUNTRY`` or ``COUNTRY_SUBDIVISION``
    (e.g. 'CR', 'US_MA' for U.S. + Massachusetts). Returns None when the
    region is None, unknown, or the 'holidays' package isn't installed.
    """
    if not region:
        return None
    try:
        import holidays
    except Exception:
        return None

    country, _, subdiv = region.upper().partition("_")
    try:
        return holidays.country_holidays(country, subdiv=subdiv or None)
    except Exception:
        return None

def _to_local(dt: datetime, tz: str) -> datetime:
    """Ensure timezone-aware datetime localized to tz."""
    tzinfo = zoneinfo.ZoneInfo(tz)
    if dt.tzinfo is None:
        # assume input is UTC if naive (common for feeds); adjust if your code stores local
        return dt.replace(tzinfo=zoneinfo.ZoneInfo("UTC")).astimezone(tzinfo)
    return dt.astimezone(tzinfo)

def _tod_bin(hour: int) -> str:
    """
    Map hour→time-of-day bin.
    Spec requires: 'morning' | 'midday' | 'afternoon' | 'evening'.
    We bucket:
      05–09 → morning
      10–13 → midday
      14–17 → afternoon
      18–04 → evening  (covers late night/overnight to keep labels strict)
    """
    if 5 <= hour <= 9:
        return "morning"
    if 10 <= hour <= 13:
        return "midday"
    if 14 <= hour <= 17:
        return "afternoon"
    return "evening"

def extract_temporal_features(
    timestamp: datetime,
    *,
    tz: str = "America/New_York",   # MBTA default
    region: str = "US_MA",          # MBTA default (US + Massachusetts)
) -> Dict[str, object]:
    """
    Returns:
        - hour: 0-23
        - day_of_week: 0-6 (Monday=0)
        - is_weekend: bool
        - is_holiday: bool (US-MA by default; set region='CR' for Costa Rica)
        - time_of_day_bin: 'morning'|'midday'|'afternoon'|'evening'
        - is_peak_hour: bool (7-9am, 4-7pm; weekdays only)
    """
    dt_local = _to_local(timestamp, tz)
    hour = dt_local.hour
    dow = dt_local.weekday()  # Monday=0
    is_weekend = dow >= 5

    cal = _get_holiday_calendar(region)
    # Holidays lib checks by date()
    is_holiday = bool(cal and (dt_local.date() in cal))

    # Peak windows (commuter assumption), weekdays only
    is_peak_hour = (dow < 5) and ((7 <= hour <= 9) or (16 <= hour <= 19))

    return {
        "hour": hour,
        "day_of_week": dow,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "time_of_day_bin": _tod_bin(hour),
        "is_peak_hour": is_peak_hour,
    }
