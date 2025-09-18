# Portfolio API - Kalshi Python SDK Documentation

## Base URL
https://api.elections.kalshi.com/trade-api/v2

## Available Methods

### Order Management

#### amend_order
```python
AmendOrderResponse amend_order(order_id, amend_order_request)
```
**Description:** Amend an existing order

**Parameters:**
- `order_id` (str): Order ID
- `amend_order_request`: Amendment request details

**Returns:** [AmendOrderResponse](https://docs.kalshi.com/python-sdk/models/AmendOrderResponse)

**HTTP Response Codes:**
- 200: Order amended successfully
- 400: Bad request - invalid input
- 401: Unauthorized - authentication required
- 404: Resource not found
- 500: Internal server error

---

#### batch_cancel_orders
```python
BatchCancelOrdersResponse batch_cancel_orders(batch_cancel_orders_request)
```
**Description:** Cancel multiple orders in a single request

**Parameters:**
- `batch_cancel_orders_request`: Batch cancellation request details

**Returns:** [BatchCancelOrdersResponse](https://docs.kalshi.com/python-sdk/models/BatchCancelOrdersResponse)

**HTTP Response Codes:**
- 200: Batch order cancellation completed
- 400: Bad request - invalid input
- 401: Unauthorized - authentication required
- 500: Internal server error

---

#### batch_create_orders
```python
BatchCreateOrdersResponse batch_create_orders(batch_create_orders_request)
```
**Description:** Create multiple orders in a single request

**Parameters:**
- `batch_create_orders_request`: Batch creation request details

**Returns:** [BatchCreateOrdersResponse](https://docs.kalshi.com/python-sdk/models/BatchCreateOrdersResponse)

**HTTP Response Codes:**
- 200: Batch order creation completed
- 400: Bad request - invalid input
- 401: Unauthorized - authentication required
- 500: Internal server error

---

#### cancel_order
```python
CancelOrderResponse cancel_order(order_id)
```
**Description:** Cancel an order

**Parameters:**
- `order_id` (str): Order ID

**Returns:** [CancelOrderResponse](https://docs.kalshi.com/python-sdk/models/CancelOrderResponse)

**HTTP Response Codes:**
- 200: Order cancelled successfully
- 401: Unauthorized - authentication required
- 404: Resource not found
- 500: Internal server error

---

#### create_order
```python
CreateOrderResponse create_order(create_order_request)
```
**Description:** Create a new order

**Parameters:**
- `create_order_request`: Order creation request details

**Returns:** [CreateOrderResponse](https://docs.kalshi.com/python-sdk/models/CreateOrderResponse)

**HTTP Response Codes:**
- 201: Order created successfully
- 400: Bad request - invalid input
- 401: Unauthorized - authentication required
- 429: Too Many Requests - rate limit exceeded
- 500: Internal server error

---

#### create_order_group
```python
CreateOrderGroupResponse create_order_group(create_order_group_request)
```
**Description:** Create a new order group

**Parameters:**
- `create_order_group_request`: Order group creation request details

**Returns:** [CreateOrderGroupResponse](https://docs.kalshi.com/python-sdk/models/CreateOrderGroupResponse)

**HTTP Response Codes:**
- 201: Order group created successfully
- 400: Bad request - invalid input
- 401: Unauthorized - authentication required
- 500: Internal server error

---

#### decrease_order
```python
DecreaseOrderResponse decrease_order(order_id, decrease_order_request)
```
**Description:** Decrease the size of an existing order

**Parameters:**
- `order_id` (str): Order ID
- `decrease_order_request`: Decrease request details

**Returns:** [DecreaseOrderResponse](https://docs.kalshi.com/python-sdk/models/DecreaseOrderResponse)

**HTTP Response Codes:**
- 200: Order decreased successfully
- 400: Bad request - invalid input
- 401: Unauthorized - authentication required
- 404: Resource not found
- 500: Internal server error

---

#### delete_order_group
```python
delete_order_group(order_group_id)
```
**Description:** Delete an order group

**Parameters:**
- `order_group_id` (str): Order group ID

**Returns:** void (empty response body)

**HTTP Response Codes:**
- 204: Order group deleted successfully
- 401: Unauthorized - authentication required
- 404: Resource not found
- 500: Internal server error

---

### Account Information

#### get_balance
```python
GetBalanceResponse get_balance()
```
**Description:** Get the user's current balance

**Parameters:** None

**Returns:** [GetBalanceResponse](https://docs.kalshi.com/python-sdk/models/GetBalanceResponse)

**HTTP Response Codes:**
- 200: Balance retrieved successfully
- 401: Unauthorized - authentication required
- 500: Internal server error

---

#### get_total_resting_order_value
```python
GetTotalRestingOrderValueResponse get_total_resting_order_value()
```
**Description:** Get the total value of all resting orders

**Parameters:** None

**Returns:** [GetTotalRestingOrderValueResponse](https://docs.kalshi.com/python-sdk/models/GetTotalRestingOrderValueResponse)

**HTTP Response Codes:**
- 200: Total resting order value retrieved successfully
- 401: Unauthorized - authentication required
- 500: Internal server error

---

### Trade Data

#### get_fills
```python
GetFillsResponse get_fills(ticker=ticker, order_id=order_id, min_ts=min_ts, max_ts=max_ts, limit=limit, cursor=cursor)
```
**Description:** Get fills for the logged-in user. A fill represents a partial or complete execution of an order. When an order matches with another order in the orderbook, a fill is created for each side of the trade.

**Parameters:**
- `ticker` (str, optional): Filter by market ticker
- `order_id` (str, optional): Filter by order ID
- `min_ts` (int, optional): Filter items after this Unix timestamp
- `max_ts` (int, optional): Filter items before this Unix timestamp
- `limit` (int, optional): Number of results per page. Defaults to 100. Maximum value is 200.
- `cursor` (str, optional): Pagination cursor

**Returns:** [GetFillsResponse](https://docs.kalshi.com/python-sdk/models/GetFillsResponse)

**HTTP Response Codes:**
- 200: Fills retrieved successfully
- 400: Bad request - invalid input
- 401: Unauthorized - authentication required
- 500: Internal server error

---

#### get_order
```python
GetOrderResponse get_order(order_id)
```
**Description:** Get a single order by ID

**Parameters:**
- `order_id` (str): Order ID

**Returns:** [GetOrderResponse](https://docs.kalshi.com/python-sdk/models/GetOrderResponse)

**HTTP Response Codes:**
- 200: Order retrieved successfully
- 401: Unauthorized - authentication required
- 404: Resource not found
- 500: Internal server error

---

#### get_order_group
```python
GetOrderGroupResponse get_order_group(order_group_id)
```
**Description:** Get details of a specific order group

**Parameters:**
- `order_group_id` (str): Order group ID

**Returns:** [GetOrderGroupResponse](https://docs.kalshi.com/python-sdk/models/GetOrderGroupResponse)

**HTTP Response Codes:**
- 200: Order group retrieved successfully
- 401: Unauthorized - authentication required
- 404: Resource not found
- 500: Internal server error

---

#### get_order_groups
```python
GetOrderGroupsResponse get_order_groups(status=status, limit=limit, cursor=cursor)
```
**Description:** Get order groups for the logged-in user

**Parameters:**
- `status` (str, optional): Filter by status
- `limit` (int, optional): Number of results per page. Defaults to 100. Maximum value is 200.
- `cursor` (str, optional): Pagination cursor

**Returns:** [GetOrderGroupsResponse](https://docs.kalshi.com/python-sdk/models/GetOrderGroupsResponse)

**HTTP Response Codes:**
- 200: Order groups retrieved successfully
- 400: Bad request - invalid input
- 401: Unauthorized - authentication required
- 500: Internal server error

---

#### get_order_queue_position
```python
GetOrderQueuePositionResponse get_order_queue_position(order_id)
```
**Description:** Get the queue position for an order

**Parameters:**
- `order_id` (str): Order ID

**Returns:** [GetOrderQueuePositionResponse](https://docs.kalshi.com/python-sdk/models/GetOrderQueuePositionResponse)

**HTTP Response Codes:**
- 200: Queue position retrieved successfully
- 401: Unauthorized - authentication required
- 404: Resource not found
- 500: Internal server error

---

#### get_orders
```python
GetOrdersResponse get_orders(ticker=ticker, event_ticker=event_ticker, min_ts=min_ts, max_ts=max_ts, status=status, limit=limit, cursor=cursor)
```
**Description:** Get orders for the logged-in user

**Parameters:**
- `ticker` (str, optional): Filter by market ticker
- `event_ticker` (str, optional): Filter by event ticker
- `min_ts` (int, optional): Filter items after this Unix timestamp
- `max_ts` (int, optional): Filter items before this Unix timestamp
- `status` (str, optional): Filter by status
- `limit` (int, optional): Number of results per page. Defaults to 100. Maximum value is 200.
- `cursor` (str, optional): Pagination cursor

**Returns:** [GetOrdersResponse](https://docs.kalshi.com/python-sdk/models/GetOrdersResponse)

**HTTP Response Codes:**
- 200: Orders retrieved successfully
- 400: Bad request - invalid input
- 401: Unauthorized - authentication required
- 500: Internal server error

---

#### get_positions
```python
GetPositionsResponse get_positions(ticker=ticker, event_ticker=event_ticker, count_down=count_down, count_up=count_up, limit=limit, cursor=cursor)
```
**Description:** Get positions for the logged-in user

**Parameters:**
- `ticker` (str, optional): Filter by market ticker
- `event_ticker` (str, optional): Filter by event ticker
- `count_down` (int, optional): Filter positions by minimum count down value
- `count_up` (int, optional): Filter positions by minimum count up value
- `limit` (int, optional): Number of results per page. Defaults to 100. Maximum value is 200.
- `cursor` (str, optional): Pagination cursor

**Returns:** [GetPositionsResponse](https://docs.kalshi.com/python-sdk/models/GetPositionsResponse)

**HTTP Response Codes:**
- 200: Positions retrieved successfully
- 400: Bad request - invalid input
- 401: Unauthorized - authentication required
- 500: Internal server error

---

#### get_queue_positions
```python
GetQueuePositionsResponse get_queue_positions(get_queue_positions_request)
```
**Description:** Get queue positions for multiple orders

**Parameters:**
- `get_queue_positions_request`: Queue positions request details

**Returns:** [GetQueuePositionsResponse](https://docs.kalshi.com/python-sdk/models/GetQueuePositionsResponse)

**HTTP Response Codes:**
- 200: Queue positions retrieved successfully
- 400: Bad request - invalid input
- 401: Unauthorized - authentication required
- 500: Internal server error

---

#### get_settlements
```python
GetSettlementsResponse get_settlements(limit=limit, cursor=cursor)
```
**Description:** Get settlements for the logged-in user

**Parameters:**
- `limit` (int, optional): Number of results per page. Defaults to 100. Maximum value is 200.
- `cursor` (str, optional): Pagination cursor

**Returns:** [GetSettlementsResponse](https://docs.kalshi.com/python-sdk/models/GetSettlementsResponse)

**HTTP Response Codes:**
- 200: Settlements retrieved successfully
- 400: Bad request - invalid input
- 401: Unauthorized - authentication required
- 500: Internal server error

---

#### reset_order_group
```python
reset_order_group(order_group_id)
```
**Description:** Reset an order group

**Parameters:**
- `order_group_id` (str): Order group ID

**Returns:** void (empty response body)

**HTTP Response Codes:**
- 204: Order group reset successfully
- 401: Unauthorized - authentication required
- 404: Resource not found
- 500: Internal server error

**Source:** https://docs.kalshi.com/python-sdk/api/PortfolioApi
