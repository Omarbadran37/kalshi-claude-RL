# Communications API - Kalshi Python SDK Documentation

Python SDK methods for Communications operations

## Available Methods

### Quote Management

#### accept_quote
```python
accept_quote(quote_id, accept_quote_request)
```
**Description:** Accept a quote. This will require the quoter to confirm

**Parameters:**
- `quote_id` (str): Quote ID
- `accept_quote_request`: Quote acceptance request details

**Returns:** None

**HTTP Response Codes:**
- 204: Quote accepted successfully
- 400: Bad request - invalid input
- 401: Unauthorized - authentication required
- 404: Resource not found
- 500: Internal server error

---

#### confirm_quote
```python
confirm_quote(quote_id)
```
**Description:** Confirm a quote. This will start a timer for order execution

**Parameters:**
- `quote_id` (str): Quote ID

**Returns:** None

**HTTP Response Codes:**
- 204: Quote confirmed successfully
- 401: Unauthorized - authentication required
- 404: Resource not found
- 500: Internal server error

---

#### create_quote
```python
CreateQuoteResponse create_quote(create_quote_request)
```
**Description:** Create a quote in response to an RFQ

**Parameters:**
- `create_quote_request`: Quote creation request details

**Returns:** [CreateQuoteResponse](https://docs.kalshi.com/python-sdk/models/CreateQuoteResponse)

**HTTP Response Codes:**
- 201: Quote created successfully
- 400: Bad request - invalid input
- 401: Unauthorized - authentication required
- 500: Internal server error

---

#### delete_quote
```python
delete_quote(quote_id)
```
**Description:** Delete a quote, which means it can no longer be accepted

**Parameters:**
- `quote_id` (str): Quote ID

**Returns:** None

**HTTP Response Codes:**
- 204: Quote deleted successfully
- 401: Unauthorized - authentication required
- 404: Resource not found
- 500: Internal server error

---

#### get_quote
```python
GetQuoteResponse get_quote(quote_id)
```
**Description:** Get a particular quote by ID

**Parameters:**
- `quote_id` (str): Quote ID

**Returns:** [GetQuoteResponse](https://docs.kalshi.com/python-sdk/models/GetQuoteResponse)

**HTTP Response Codes:**
- 200: Quote retrieved successfully
- 401: Unauthorized - authentication required
- 404: Resource not found
- 500: Internal server error

---

#### get_quotes
```python
GetQuotesResponse get_quotes()
```
**Description:** Retrieve all quotes

**Parameters:** None

**Returns:** [GetQuotesResponse](https://docs.kalshi.com/python-sdk/models/GetQuotesResponse)

**HTTP Response Codes:**
- 200: Quotes retrieved successfully
- 401: Unauthorized - authentication required
- 500: Internal server error

---

### RFQ (Request for Quote) Management

#### create_rfq
```python
CreateRFQResponse create_rfq(create_rfq_request)
```
**Description:** Create a new RFQ

**Parameters:**
- `create_rfq_request`: RFQ creation request details

**Returns:** [CreateRFQResponse](https://docs.kalshi.com/python-sdk/models/CreateRFQResponse)

**HTTP Response Codes:**
- 201: RFQ created successfully
- 400: Bad request - invalid input
- 401: Unauthorized - authentication required
- 500: Internal server error

---

#### delete_rfq
```python
delete_rfq(rfq_id)
```
**Description:** Delete an RFQ by ID

**Parameters:**
- `rfq_id` (str): RFQ ID

**Returns:** None

**HTTP Response Codes:**
- 204: RFQ deleted successfully
- 401: Unauthorized - authentication required
- 404: Resource not found
- 500: Internal server error

---

#### get_rfq
```python
GetRFQResponse get_rfq(rfq_id)
```
**Description:** Get a single RFQ by ID

**Parameters:**
- `rfq_id` (str): RFQ ID

**Returns:** [GetRFQResponse](https://docs.kalshi.com/python-sdk/models/GetRFQResponse)

**HTTP Response Codes:**
- 200: RFQ retrieved successfully
- 401: Unauthorized - authentication required
- 404: Resource not found
- 500: Internal server error

---

#### get_rfqs
```python
GetRFQsResponse get_rfqs()
```
**Description:** Retrieve all RFQs

**Parameters:** None

**Returns:** [GetRFQsResponse](https://docs.kalshi.com/python-sdk/models/GetRFQsResponse)

**HTTP Response Codes:**
- 200: RFQs retrieved successfully
- 401: Unauthorized - authentication required
- 500: Internal server error

---

### User Information

#### get_communications_id
```python
GetCommunicationsIDResponse get_communications_id()
```
**Description:** Get the communications ID of the logged-in user

**Parameters:** None

**Returns:** [GetCommunicationsIDResponse](https://docs.kalshi.com/python-sdk/models/GetCommunicationsIDResponse)

**HTTP Response Codes:**
- 200: Communications ID retrieved successfully
- 401: Unauthorized - authentication required
- 500: Internal server error

**Source:** https://docs.kalshi.com/python-sdk/api/CommunicationsApi
