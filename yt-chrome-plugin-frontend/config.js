// Settings shared by the popup, options page, background worker and content script.
const DEFAULT_SETTINGS = { apiUrl: 'http://localhost:8080', showBadges: true };

async function getSettings() {
  const settings = await chrome.storage.sync.get(DEFAULT_SETTINGS);
  return { ...settings, apiUrl: settings.apiUrl.replace(/\/+$/, '') };
}
