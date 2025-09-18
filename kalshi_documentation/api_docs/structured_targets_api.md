# Structured Targets API - Kalshi Python SDK Documentation

## Base URL
https://api.elections.kalshi.com/trade-api/v2

## Available Methods

### Structured Target Management

#### get_structured_target
```python
GetStructuredTargetResponse get_structured_target(structured_target_id)
```
**Description:** Get a single structured target by ID

**Parameters:**
- `structured_target_id` (str): Structured target ID

**Returns:** [GetStructuredTargetResponse](https://docs.kalshi.com/python-sdk/models/GetStructuredTargetResponse)

**HTTP Response Codes:**
- 200: Structured target retrieved successfully
- 401: Unauthorized - authentication required
- 404: Resource not found
- 500: Internal server error

---

#### get_structured_targets
```python
GetStructuredTargetsResponse get_structured_targets(status=status, page_size=page_size)
```
**Description:** Get all structured targets

**Parameters:**
- `status` (str, optional): Filter by structured target status
- `page_size` (int, optional): Number of items per page (minimum 100, default 100)

**Returns:** [GetStructuredTargetsResponse](https://docs.kalshi.com/python-sdk/models/GetStructuredTargetsResponse)

**HTTP Response Codes:**
- 200: Structured targets retrieved successfully
- 401: Unauthorized - authentication required
- 500: Internal server error

**Source:** https://docs.kalshi.com/python-sdk/api/StructuredTargetsApi
