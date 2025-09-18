# Exchange API - Kalshi Python SDK Documentation

## Base URL
https://api.elections.kalshi.com/trade-api/v2

## Available Methods

### Exchange Information

#### get_exchange_announcements
```python
GetExchangeAnnouncementsResponse get_exchange_announcements()
```
**Description:** Get all exchange-wide announcements

**Parameters:** None

**Returns:** [GetExchangeAnnouncementsResponse](https://docs.kalshi.com/python-sdk/models/GetExchangeAnnouncementsResponse)

**HTTP Response Codes:**
- 200: Announcements retrieved successfully
- 500: Internal server error

---

#### get_exchange_schedule
```python
GetExchangeScheduleResponse get_exchange_schedule()
```
**Description:** Get the exchange schedule

**Parameters:** None

**Returns:** [GetExchangeScheduleResponse](https://docs.kalshi.com/python-sdk/models/GetExchangeScheduleResponse)

**HTTP Response Codes:**
- 200: Schedule retrieved successfully
- 500: Internal server error

---

#### get_exchange_status
```python
ExchangeStatus get_exchange_status()
```
**Description:** Get the exchange status

**Parameters:** None

**Returns:** [ExchangeStatus](https://docs.kalshi.com/python-sdk/models/ExchangeStatus)

**HTTP Response Codes:**
- 200: Status retrieved successfully
- 500: Internal server error

---

#### get_user_data_timestamp
```python
GetUserDataTimestampResponse get_user_data_timestamp()
```
**Description:** There is typically a short delay before exchange events are reflected in the API endpoints. Whenever possible, combine API responses to PUT/POST/DELETE requests with websocket data to obtain the most accurate view of the exchange state. This endpoint provides an approximate indication of when the data from the following endpoints was last validated: GetBalance, GetOrder(s), GetFills, GetPositions

**Parameters:** None

**Returns:** [GetUserDataTimestampResponse](https://docs.kalshi.com/python-sdk/models/GetUserDataTimestampResponse)

**HTTP Response Codes:**
- 200: Timestamp retrieved successfully
- 401: Unauthorized - authentication required
- 500: Internal server error

**Source:** https://docs.kalshi.com/python-sdk/api/ExchangeApi
