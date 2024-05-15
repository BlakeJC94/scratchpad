from copy import copy
from dataclasses import asdict, dataclass, field
from typing import Any

from quakerv2.file import FILE_FMTS
from quakerv2.utils import check_time_field_is_valid


@dataclass
class Query:
    format: str = field(default=None)
    # Time
    endtime: str = field(default=None)
    starttime: str = field(default=None)
    updatedafter: str = field(default=None)
    # Location (rectangle)
    minlatitude: float = field(default=None)
    maxlatitude: float = field(default=None)
    minlongitude: float = field(default=None)
    maxlongitude: float = field(default=None)
    # Location (circle)
    latitude: float = field(default=None)
    longitude: float = field(default=None)
    maxradius: float = field(default=None)
    maxradiuskm: float = field(default=None)
    # Other
    catalog: str = field(default=None)
    contributor: str = field(default=None)
    eventid: str = field(default=None)
    includeallmagnitudes: bool = field(default=None)
    includeallorigins: bool = field(default=None)
    includearrivals: bool = field(default=None)
    includedeleted: bool = field(default=None)
    includesuperseded: bool = field(default=None)
    limit: int = field(default=None)
    maxdepth: float = field(default=None)
    maxmagnitude: float = field(default=None)
    mindepth: float = field(default=None)
    minmagnitude: float = field(default=None)
    offset: int = field(default=None)
    orderby: str = field(default=None)
    # Extensions
    alertlevel: str = field(default=None)
    callback: str = field(default=None)
    eventtype: str = field(default=None)
    jsonerror: bool = field(default=None)
    kmlanimated: bool = field(default=None)
    kmlcolorby: str = field(default=None)
    maxcdi: float = field(default=None)
    maxgap: float = field(default=None)
    maxmmi: float = field(default=None)
    maxsig: int = field(default=None)
    mincdi: float = field(default=None)
    minfelt: int = field(default=None)
    mingap: float = field(default=None)
    minsig: int = field(default=None)
    producttype: str = field(default=None)
    productcode: str = field(default=None)
    reviewstatus: str = field(default=None)

    def __post_init__(self):
        self.validate()

    def validate(self):
        for time_field in ["starttime", "endtime", "updatedafter"]:
            if (time := getattr(self, time_field)) is not None:
                check_time_field_is_valid(time)
        if not any(
            v == self.orderby for v in [None, "time", "time-asc", "magnitude", "magnitude-asc"]
        ):
            raise ValueError(f"Invalid orderby value: '{self.orderby}'.")
        if not any(v == self.format for v in FILE_FMTS.keys()):
            raise ValueError(f"Invalid format value: '{self.format}'.")

    def copy(self):
        return copy(self)

    def dict(self, include_nones: bool = False) -> dict[str, Any]:
        query_dict = asdict(self)
        if include_nones:
            return query_dict
        return {k: v for k, v in query_dict.items() if v is not None}
