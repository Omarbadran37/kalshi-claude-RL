# Series API - Kalshi Python SDK Documentation

## Base URL
https://api.elections.kalshi.com/trade-api/v2

## Available Methods

### Series Information

#### get_series
```python
GetSeriesResponse get_series(status=status)
```
**Description:** Get all market series

**Parameters:**
- `status` (str, optional): Filter by series status

**Returns:** [GetSeriesResponse](https://docs.kalshi.com/python-sdk/models/GetSeriesResponse)

**HTTP Response Codes:**
- 200: Series retrieved successfully
- 401: Unauthorized - authentication required
- 500: Internal server error

---

#### get_series_by_ticker
```python
GetSeriesByTickerResponse get_series_by_ticker(ticker)
```
**Description:** Get a single series by its ticker

**Parameters:**
- `ticker` (str): The series ticker

**Returns:** [GetSeriesByTickerResponse](https://docs.kalshi.com/python-sdk/models/GetSeriesByTickerResponse)

**HTTP Response Codes:**
- 200: Series retrieved successfully
- 401: Unauthorized - authentication required
- 404: Resource not found
- 500: Internal server error

**Source:** https://docs.kalshi.com/python-sdk/api/SeriesApi
