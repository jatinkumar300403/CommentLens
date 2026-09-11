// Options page: choose the API URL and toggle the in-page comment badges.
const apiUrlInput = document.getElementById('api-url');
const showBadgesInput = document.getElementById('show-badges');
const status = document.getElementById('status');

function setStatus(message, isError = false) {
  status.textContent = message;
  status.classList.toggle('error', isError);
}

function parseOrigin(value) {
  let url;
  try {
    url = new URL(value.trim());
  } catch {
    return null;
  }
  return ['http:', 'https:'].includes(url.protocol) ? url.origin : null;
}

document.addEventListener('DOMContentLoaded', async () => {
  const { apiUrl, showBadges } = await getSettings();
  apiUrlInput.value = apiUrl;
  showBadgesInput.checked = showBadges;
});

document.getElementById('save').addEventListener('click', async () => {
  const origin = parseOrigin(apiUrlInput.value);
  if (!origin) {
    setStatus('Enter a valid http(s) URL, like http://localhost:8080', true);
    return;
  }

  // The extension can only call hosts it has permission for; Chrome asks the user for new ones on click
  const granted = await chrome.permissions.request({ origins: [`${origin}/*`] });
  if (!granted) {
    setStatus(`Permission to reach ${origin} was not granted.`, true);
    return;
  }

  await chrome.storage.sync.set({ apiUrl: origin, showBadges: showBadgesInput.checked });
  setStatus('Saved.');
});

document.getElementById('test').addEventListener('click', async () => {
  const origin = parseOrigin(apiUrlInput.value);
  if (!origin) {
    setStatus('Enter a valid http(s) URL, like http://localhost:8080', true);
    return;
  }

  try {
    const response = await fetch(`${origin}/health`);
    const body = await response.json();
    if (body.youtube_api_key_configured) setStatus('Connected: the API is healthy.');
    else setStatus('Connected, but YOUTUBE_API_KEY is not set on the server.', true);
  } catch {
    setStatus(`Cannot reach ${origin}. Check that the API is running.`, true);
  }
});
