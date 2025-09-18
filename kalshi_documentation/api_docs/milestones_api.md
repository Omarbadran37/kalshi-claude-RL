# Milestones API - Kalshi Python SDK Documentation

## Base URL
https://api.elections.kalshi.com/trade-api/v2

## Available Methods

### Milestone Information

#### get_milestone
```python
GetMilestoneResponse get_milestone(milestone_id)
```
**Description:** Get a single milestone by ID

**Parameters:**
- `milestone_id` (str): Milestone ID

**Returns:** [GetMilestoneResponse](https://docs.kalshi.com/python-sdk/models/GetMilestoneResponse)

**HTTP Response Codes:**
- 200: Milestone retrieved successfully
- 401: Unauthorized - authentication required
- 404: Resource not found
- 500: Internal server error

---

#### get_milestones
```python
GetMilestonesResponse get_milestones(status=status, limit=limit)
```
**Description:** Get all milestones

**Parameters:**
- `status` (str, optional): Filter by milestone status
- `limit` (int, optional): Number of items per page (minimum 1, maximum 500) (default: 100)

**Returns:** [GetMilestonesResponse](https://docs.kalshi.com/python-sdk/models/GetMilestonesResponse)

**HTTP Response Codes:**
- 200: Milestones retrieved successfully
- 401: Unauthorized - authentication required
- 500: Internal server error

**Source:** https://docs.kalshi.com/python-sdk/api/MilestonesApi
