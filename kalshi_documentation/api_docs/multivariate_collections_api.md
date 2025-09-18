# Multivariate Collections API - Kalshi Python SDK Documentation

## Base URL
https://api.elections.kalshi.com/trade-api/v2

## Available Methods

### Multivariate Event Collection Management

#### get_multivariate_event_collection
```python
GetMultivariateEventCollectionResponse get_multivariate_event_collection(collection_ticker)
```
**Description:** Get a single multivariate event collection by ticker

**Parameters:**
- `collection_ticker` (str): Collection ticker

**Returns:** [GetMultivariateEventCollectionResponse](https://docs.kalshi.com/python-sdk/models/GetMultivariateEventCollectionResponse)

**HTTP Response Codes:**
- 200: Collection retrieved successfully
- 401: Unauthorized - authentication required
- 404: Resource not found
- 500: Internal server error

---

#### get_multivariate_event_collections
```python
GetMultivariateEventCollectionsResponse get_multivariate_event_collections(status=status)
```
**Description:** Get all multivariate event collections

**Parameters:**
- `status` (str, optional): Filter by multivariate collection status

**Returns:** [GetMultivariateEventCollectionsResponse](https://docs.kalshi.com/python-sdk/models/GetMultivariateEventCollectionsResponse)

**HTTP Response Codes:**
- 200: Collections retrieved successfully
- 401: Unauthorized - authentication required
- 500: Internal server error

---

#### lookup_multivariate_event_collection_bundle
```python
LookupBundleResponse lookup_multivariate_event_collection_bundle(collection_ticker, lookup_bundle_request)
```
**Description:** Lookup a bundle in a multivariate event collection

**Parameters:**
- `collection_ticker` (str): Collection ticker
- `lookup_bundle_request`: Bundle lookup request details

**Returns:** [LookupBundleResponse](https://docs.kalshi.com/python-sdk/models/LookupBundleResponse)

**HTTP Response Codes:**
- 200: Bundle lookup successful
- 400: Bad request - invalid input
- 401: Unauthorized - authentication required
- 404: Resource not found
- 500: Internal server error

**Source:** https://docs.kalshi.com/python-sdk/api/MultivariateCollectionsApi
