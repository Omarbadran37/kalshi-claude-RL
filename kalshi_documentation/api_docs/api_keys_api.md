# API Keys API - Kalshi Python SDK Documentation

## Base URL
https://api.elections.kalshi.com/trade-api/v2

## Available Methods

### API Key Management

#### create_api_key
```python
CreateApiKeyResponse create_api_key(create_api_key_request)
```
**Description:** Create a new API key with a user-provided public key. This endpoint allows users with Premier or Market Maker API usage levels to create API keys by providing their own RSA public key. The platform will use this public key to verify signatures on API requests.

**Parameters:**
- `create_api_key_request`: API key creation request details

**Returns:** [CreateApiKeyResponse](https://docs.kalshi.com/python-sdk/models/CreateApiKeyResponse)

**HTTP Response Codes:**
- 201: API key created successfully
- 400: Bad request - invalid input
- 401: Unauthorized - authentication required
- 403: Forbidden - insufficient permissions
- 500: Internal server error

---

#### delete_api_key
```python
delete_api_key(api_key)
```
**Description:** Delete an existing API key. This endpoint permanently deletes an API key. Once deleted, the key can no longer be used for authentication. This action cannot be undone.

**Parameters:**
- `api_key` (str): API key ID to delete

**Returns:** void (empty response body)

**HTTP Response Codes:**
- 204: API key successfully deleted
- 400: Bad request - invalid input
- 401: Unauthorized - authentication required
- 404: Resource not found
- 500: Internal server error

---

#### generate_api_key
```python
GenerateApiKeyResponse generate_api_key(generate_api_key_request)
```
**Description:** Generate a new API key with an automatically created key pair. This endpoint generates both a public and private RSA key pair. The public key is stored on the platform, while the private key is returned to the user and must be stored securely. The private key cannot be retrieved again.

**Parameters:**
- `generate_api_key_request`: API key generation request details

**Returns:** [GenerateApiKeyResponse](https://docs.kalshi.com/python-sdk/models/GenerateApiKeyResponse)

**HTTP Response Codes:**
- 201: API key generated successfully
- 400: Bad request - invalid input
- 401: Unauthorized - authentication required
- 500: Internal server error

---

#### get_api_keys
```python
GetApiKeysResponse get_api_keys()
```
**Description:** Retrieve all API keys associated with the authenticated user. API keys allow programmatic access to the platform without requiring username/password authentication. Each key has a unique identifier and name.

**Parameters:** None

**Returns:** [GetApiKeysResponse](https://docs.kalshi.com/python-sdk/models/GetApiKeysResponse)

**HTTP Response Codes:**
- 200: List of API keys retrieved successfully
- 401: Unauthorized - authentication required
- 500: Internal server error

**Source:** https://docs.kalshi.com/python-sdk/api/ApiKeysApi
