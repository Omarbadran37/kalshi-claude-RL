# Events API - Kalshi Python SDK Documentation

## Base URL
https://api.elections.kalshi.com/trade-api/v2

## Available Methods

### Event Information

#### get_event
```python
GetEventResponse get_event(event_ticker, with_nested_markets=with_nested_markets)
```
**Description:** Get data about an event by its ticker. An event represents a real-world occurrence that can be traded on, such as an election, sports game, or economic indicator release. Events contain one or more markets where users can place trades on different outcomes.

**Parameters:**
- `event_ticker` (str): Event ticker
- `with_nested_markets` (bool, optional): If true, markets are included within the event object. If false (default), markets are returned as a separate top-level field in the response. (default: False)

**Returns:** [GetEventResponse](https://docs.kalshi.com/python-sdk/models/GetEventResponse)

**HTTP Response Codes:**
- 200: Event retrieved successfully
- 401: Unauthorized - authentication required
- 404: Resource not found
- 500: Internal server error

---

#### get_event_metadata
```python
GetEventMetadataResponse get_event_metadata(event_ticker)
```
**Description:** Get metadata about an event by its ticker

**Parameters:**
- `event_ticker` (str): Event ticker

**Returns:** [GetEventMetadataResponse](https://docs.kalshi.com/python-sdk/models/GetEventMetadataResponse)

**HTTP Response Codes:**
- 200: Event metadata retrieved successfully
- 401: Unauthorized - authentication required
- 404: Resource not found
- 500: Internal server error

---

#### get_events
```python
GetEventsResponse get_events(limit=limit, cursor=cursor, with_nested_markets=with_nested_markets, status=status, series_ticker=series_ticker, min_close_ts=min_close_ts)
```
**Description:** Get data about all events. An event represents a real-world occurrence that can be traded on, such as an election, sports game, or economic indicator release. Events contain one or more markets where users can place trades on different outcomes. This endpoint returns a paginated response. Use the 'limit' parameter to control page size (1-200, defaults to 100). The response includes a 'cursor' field - pass this value in the 'cursor' parameter of your next request to get the next page. An empty cursor indicates no more pages are available.

**Parameters:**
- `limit` (int, optional): Number of results per page. Defaults to 100. Maximum value is 200.
- `cursor` (str, optional): Pagination cursor. Use the cursor value returned from the previous response to get the next page of results. Leave empty for the first page.
- `with_nested_markets` (bool, optional): If true, markets are included within the event object. If false (default), markets are returned as a separate top-level field in the response. (default: False)
- `status` (str, optional): Filter by status. Possible values depend on the endpoint.
- `series_ticker` (str, optional): Filter by series ticker
- `min_close_ts` (int, optional): Filter items that close after this Unix timestamp

**Returns:** [GetEventsResponse](https://docs.kalshi.com/python-sdk/models/GetEventsResponse)

**HTTP Response Codes:**
- 200: Events retrieved successfully
- 400: Bad request - invalid input
- 401: Unauthorized - authentication required
- 500: Internal server error

**Source:** https://docs.kalshi.com/python-sdk/api/EventsApi
