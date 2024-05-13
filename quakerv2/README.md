# quakerv2

This is an exploritory re-write of the application `quaker`

The original version implements the USGS API completely, and uses some tricked to break up large
queries into sequential sub-queries. I wonder if there's a way to multi-thread the requests
non-sequentially instead? This may come at the cost of restricting the request API *slightly*, but
this might be worth the trade-off.

[API documentation](https://earthquake.usgs.gov/fdsnws/event/1/)


Brief notes on thoughts:

- Will only support CSV format for now
- Will only support sorting results chronologically, not by magnitude
- Will not support reversing the order (always descending)
- Provide filters for location, magnitude
    - All requests need a time limit
        - Base dataclass
        - Can add extensions later
    - Subclass request into rectangle/circle location requests
- Processing requests
    - Split request into 12hr time windows
    - Randomly stagger request submissions
    - Join results in memory before writing to disk

