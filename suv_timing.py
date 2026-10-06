"""Conservative standard-DICOM SUVbw timing (no vendor-specific inference)."""

import math
import re
from datetime import datetime


def positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def parse_tm(value):
    text = str(value)
    if not re.fullmatch(r"\d{6}(?:\.\d{1,6})?", text):
        raise ValueError("Complete DICOM time required")
    return datetime.strptime(text, "%H%M%S.%f" if "." in text else "%H%M%S")


def parse_dt(value):
    text = str(value)
    if not re.fullmatch(r"\d{14}(?:\.\d{1,6})?(?:[+-]\d{4})?", text):
        raise ValueError("Complete DICOM datetime required")
    zone = bool(re.search(r"[+-]\d{4}$", text))
    return datetime.strptime(text, "%Y%m%d%H%M%S" + (".%f" if "." in text else "") + ("%z" if zone else ""))


def decay_seconds(ds, acquisition_time=None, *, allow_same_day_injection=False):
    """ADMIN needs no time shift; START requires an unambiguous acquisition time.

    Time-only injection requires explicit same-day confirmation, never inferred.
    Missing/malformed times, ambiguous midnight rollover and NONE are rejected.
    SeriesTime is deliberately not substituted for AcquisitionTime.
    """
    mode = str(getattr(ds, "DecayCorrection", "")).upper()
    if mode == "ADMIN":
        return 0.0
    if mode != "START":
        raise ValueError("Only START or ADMIN decay correction is supported")
    rp = ds.RadiopharmaceuticalInformationSequence[0]
    injection_dt = getattr(rp, "RadiopharmaceuticalStartDateTime", None)
    acquisition_dt = getattr(ds, "AcquisitionDateTime", None)
    if acquisition_time is not None and (injection_dt or acquisition_dt):
        raise ValueError("Time-only override cannot replace datetime metadata")
    if acquisition_dt:
        end = parse_dt(acquisition_dt)
    else:
        time = parse_tm(acquisition_time if acquisition_time is not None else getattr(ds, "AcquisitionTime", None))
        date = getattr(ds, "AcquisitionDate", None)
        end = datetime.combine(datetime.strptime(str(date), "%Y%m%d").date(), time.time()) if date else time
    if injection_dt:
        start = parse_dt(injection_dt)
        if not acquisition_dt and not getattr(ds, "AcquisitionDate", None):
            raise ValueError("Acquisition date required with injection datetime")
    else:
        if not allow_same_day_injection:
            raise ValueError('Injection date missing; explicit same-day confirmation required')
        start = datetime.combine(end.date(), parse_tm(getattr(rp, "RadiopharmaceuticalStartTime", None)).time())
        start = start.replace(tzinfo=end.tzinfo)
    if (end.tzinfo is None) != (start.tzinfo is None):
        raise ValueError('Mixed timezone-aware and naive datetimes are unsupported')
    seconds = (end - start).total_seconds()
    if seconds < 0 or not math.isfinite(seconds):
        raise ValueError("Negative or ambiguous injection-to-acquisition interval")
    return seconds
