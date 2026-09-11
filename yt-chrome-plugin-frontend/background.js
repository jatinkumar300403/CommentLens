importScripts('config.js');

// Content scripts run inside youtube.com, where page CORS and Chrome's local-network rules block
// calls to the API. They send comment text here instead; the worker calls the API with the
// extension's own host permissions.
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type !== 'predict') return false;

  predict(message.comments)
    .then((predictions) => sendResponse({ predictions }))
    .catch((error) => sendResponse({ error: error.message }));
  return true; // Keep the message channel open for the async response
});

async function predict(comments) {
  const { apiUrl } = await getSettings();
  const response = await fetch(`${apiUrl}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ comments }),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `API returned HTTP ${response.status}`);
  return result.map((item) => item.sentiment);
}
