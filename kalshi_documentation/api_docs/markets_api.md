# Markets API - Kalshi Python SDK Documentation

## Base URL
https://api.elections.kalshi.com/trade-api/v2

## Available Methods

### Market Information

#### get_market
```python
GetMarketResponse get_market(ticker)
```
**Description:** Get a single market by its ticker. A market represents a specific binary outcome within an event that users can trade on (e.g., "Will candidate X win?"). Markets have yes/no positions, current prices, volume, and settlement rules.

**Parameters:**
- `ticker` (str): Market ticker

**Returns:** [GetMarketResponse](https://docs.kalshi.com/python-sdk/models/GetMarketResponse)

**HTTP Response Codes:**
- 200: Market retrieved successfully
- 401: Unauthorized - authentication required
- 404: Resource not found
- 500: Internal server error

---

#### get_markets
```python
GetMarketsResponse get_markets(limit=limit, cursor=cursor, event_ticker=event_ticker, series_ticker=series_ticker, max_close_ts=max_close_ts, min_close_ts=min_close_ts, status=status, tickers=tickers)
```
**Description:** List and discover markets on Kalshi. A market represents a specific binary outcome within an event that users can trade on (e.g., "Will candidate X win?"). Markets have yes/no positions, current prices, volume, and settlement rules. This endpoint returns a paginated response. Use the 'limit' parameter to control page size (1-1000, defaults to 100). The response includes a 'cursor' field - pass this value in the 'cursor' parameter of your next request to get the next page. An empty cursor indicates no more pages are available.

**Parameters:**
- `limit` (int, optional): Number of results per page. Defaults to 100. Maximum value is 1000.
- `cursor` (str, optional): Pagination cursor. Use the cursor value returned from the previous response to get the next page of results. Leave empty for the first page.
- `event_ticker` (str, optional): Filter by event ticker
- `series_ticker` (str, optional): Filter by series ticker
- `max_close_ts` (int, optional): Filter items that close before this Unix timestamp
- `min_close_ts` (int, optional): Filter items that close after this Unix timestamp
- `status` (str, optional): Filter by market status. Comma-separated list. Possible values are 'initialized', 'open', 'closed', 'settled', 'determined'. Note that the API accepts 'open' for filtering but returns 'active' in the response. Leave empty to return markets with any status.
- `tickers` (str, optional): Filter by specific market tickers. Comma-separated list of market tickers to retrieve.

**Returns:** [GetMarketsResponse](https://docs.kalshi.com/python-sdk/models/GetMarketsResponse)

**HTTP Response Codes:**
- 200: Markets retrieved successfully
- 400: Bad request - invalid input
- 401: Unauthorized - authentication required
- 500: Internal server error

---

### Market Data

#### get_market_candlesticks
```python
GetMarketCandlesticksResponse get_market_candlesticks(ticker, market_ticker, start_ts=start_ts, end_ts=end_ts, period_interval=period_interval)
```
**Description:** Get candlestick data for a market within a series

**Parameters:**
- `ticker` (str): The series ticker
- `market_ticker` (str): The market ticker
- `start_ts` (int, optional): Start timestamp for the range
- `end_ts` (int, optional): End timestamp for the range
- `period_interval` (str, optional): Period interval for candlesticks (e.g., 1m, 5m, 1h, 1d)

**Returns:** [GetMarketCandlesticksResponse](https://docs.kalshi.com/python-sdk/models/GetMarketCandlesticksResponse)

**HTTP Response Codes:**
- 200: Candlesticks retrieved successfully
- 400: Bad request - invalid input
- 401: Unauthorized - authentication required
- 404: Resource not found
- 500: Internal server error

---

#### get_market_orderbook
```python
GetMarketOrderbookResponse get_market_orderbook(ticker, depth=depth)
```
**Description:** Get the orderbook for a market

**Parameters:**
- `ticker` (str): Market ticker
- `depth` (int, optional): Depth of the orderbook to retrieve (default: 10)

**Returns:** [GetMarketOrderbookResponse](https://docs.kalshi.com/python-sdk/models/GetMarketOrderbookResponse)

**HTTP Response Codes:**
- 200: Orderbook retrieved successfully
- 401: Unauthorized - authentication required
- 404: Resource not found
- 500: Internal server error

---

#### get_trades
```python
GetTradesResponse get_trades(limit=limit, cursor=cursor, ticker=ticker, min_ts=min_ts, max_ts=max_ts)
```
**Description:** Get all trades for all markets. A trade represents a completed transaction between two users on a specific market. Each trade includes the market ticker, price, quantity, and timestamp information. This endpoint returns a paginated response. Use the 'limit' parameter to control page size (1-1000, defaults to 100). The response includes a 'cursor' field - pass this value in the 'cursor' parameter of your next request to get the next page. An empty cursor indicates no more pages are available.

**Parameters:**
- `limit` (int, optional): Number of results per page. Defaults to 100. Maximum value is 1000.
- `cursor` (str, optional): Pagination cursor. Use the cursor value returned from the previous response to get the next page of results. Leave empty for the first page.
- `ticker` (str, optional): Filter by market ticker
- `min_ts` (int, optional): Filter items after this Unix timestamp
- `max_ts` (int, optional): Filter items before this Unix timestamp

**Returns:** [GetTradesResponse](https://docs.kalshi.com/python-sdk/models/GetTradesResponse)

**HTTP Response Codes:**
- 200: Trades retrieved successfully
- 400: Bad request - invalid input
- 500: Internal server error

**Source:** https://docs.kalshi.com/python-sdk/api/MarketsApi
