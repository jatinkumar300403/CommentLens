// Labels each YouTube comment in place: green = positive, blue = neutral, red = negative.
const BADGES = {
  '1': { label: '✔ Positive', className: 'yts-badge--positive' },
  '0': { label: '● Neutral', className: 'yts-badge--neutral' },
  '-1': { label: '✖ Negative', className: 'yts-badge--negative' },
};
const COMMENT_TEXT_SELECTOR = '#comments #content-text';
const BATCH_SIZE = 50;

let scheduled = null;
let running = false;

function textOf(element) {
  return element.innerText.trim();
}

function commentContainer(element) {
  return element.closest('ytd-comment-view-model, ytd-comment-renderer') || element.parentElement;
}

function removeBadge(textElement) {
  const container = commentContainer(textElement);
  container.querySelectorAll('.yts-badge').forEach((badge) => {
    if (commentContainer(badge) === container) badge.remove();
  });
}

function addBadge(textElement, sentiment) {
  const badge = BADGES[String(sentiment)];
  if (!badge) return;

  removeBadge(textElement);
  const element = document.createElement('span');
  element.className = `yts-badge ${badge.className}`;
  element.textContent = badge.label;
  element.title = 'Sentiment predicted by YouTube Sentiment Insights';

  const header = commentContainer(textElement).querySelector('#header-author');
  if (header) header.append(element);
  else textElement.before(element);
}

async function labelComments() {
  if (running || location.pathname !== '/watch') return;
  running = true;
  try {
    const { showBadges } = await getSettings();
    if (!showBadges) return;

    // dataset.ytsFor records which text was labelled: YouTube reuses comment elements across videos
    const pending = [...document.querySelectorAll(COMMENT_TEXT_SELECTOR)]
      .filter((element) => textOf(element) && element.dataset.ytsFor !== textOf(element))
      .slice(0, BATCH_SIZE);
    if (pending.length === 0) return;

    const texts = pending.map(textOf);
    pending.forEach((element, i) => { element.dataset.ytsFor = texts[i]; });

    const response = await chrome.runtime.sendMessage({ type: 'predict', comments: texts });
    if (!response || response.error) throw new Error(response?.error || 'No response from the background worker');

    response.predictions.forEach((sentiment, i) => {
      if (textOf(pending[i]) === texts[i]) addBadge(pending[i], sentiment);
    });
    scheduleLabelling(); // More comments may have loaded while waiting
  } catch (error) {
    console.warn('[YouTube Sentiment Insights] Could not label comments:', error.message);
  } finally {
    running = false;
  }
}

function scheduleLabelling() {
  if (scheduled) return;
  // Throttle rather than debounce: YouTube mutates the page constantly, so a debounce might never fire
  scheduled = setTimeout(() => {
    scheduled = null;
    labelComments();
  }, 1000);
}

function clearBadges() {
  document.querySelectorAll('.yts-badge').forEach((badge) => badge.remove());
  document.querySelectorAll('[data-yts-for]').forEach((element) => { delete element.dataset.ytsFor; });
}

new MutationObserver(scheduleLabelling).observe(document.body, { childList: true, subtree: true });

chrome.storage.onChanged.addListener((changes) => {
  if (!changes.showBadges) return;
  if (changes.showBadges.newValue) scheduleLabelling();
  else clearBadges();
});

scheduleLabelling();
