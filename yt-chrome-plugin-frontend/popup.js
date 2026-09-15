// Popup dashboard: fetch the current video's comments through the API, classify them, render the results.
const SENTIMENTS = {
  '1': { label: 'Positive', className: 'positive' },
  '0': { label: 'Neutral', className: 'neutral' },
  '-1': { label: 'Negative', className: 'negative' },
};
const TOP_N = 25;
const MAX_COMMENTS = 500;

document.addEventListener('DOMContentLoaded', async () => {
  document.getElementById('open-options').addEventListener('click', (event) => {
    event.preventDefault();
    chrome.runtime.openOptionsPage();
  });

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const videoId = videoIdFromUrl(tab?.url);
  if (!videoId) {
    setStatus('Open a YouTube video to analyse its comments.');
    return;
  }

  const { apiUrl } = await getSettings();
  try {
    setStatus('Fetching comments…');
    const query = new URLSearchParams({ video_id: videoId, max_comments: MAX_COMMENTS });
    const { comments } = await withRetry(() => requestJson(`${apiUrl}/comments?${query}`));
    if (comments.length === 0) {
      setStatus('No comments found for this video (or comments are turned off).');
      return;
    }

    setStatus(`Analysing ${comments.length} comments…`);
    const predictions = await withRetry(() => postJson(`${apiUrl}/predict_with_timestamps`, {
      comments: comments.map(({ text, timestamp }) => ({ text, timestamp })),
    }));

    const counts = renderResults(comments, predictions);
    setStatus('');
    await Promise.all([
      loadChart('pie-chart', `${apiUrl}/generate_chart`, { sentiment_counts: counts }),
      loadChart('trend-graph', `${apiUrl}/generate_trend_graph`, {
        sentiment_data: predictions.map(({ timestamp, sentiment }) => ({ timestamp, sentiment })),
      }),
      loadChart('wordcloud', `${apiUrl}/generate_wordcloud`, { comments: comments.map((c) => c.text) }),
    ]);
  } catch (error) {
    setStatus(error.message, true);
  }
});

function videoIdFromUrl(url) {
  try {
    const { hostname, pathname, searchParams } = new URL(url);
    if (!/(^|\.)youtube\.com$/.test(hostname)) return null;
    const id = pathname === '/watch' ? searchParams.get('v') : pathname.match(/^\/shorts\/([\w-]{11})/)?.[1];
    return /^[\w-]{11}$/.test(id ?? '') ? id : null;
  } catch {
    return null;
  }
}

async function requestJson(url, options = {}) {
  let response;
  try {
    response = await fetch(url, options);
  } catch {
    throw new Error(`Cannot reach the API at ${new URL(url).origin}. Is it running?`);
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(body.error || `API returned HTTP ${response.status}`);
    error.retryable = response.status === 503; // Cloud Run starting an instance, or the model still loading
    throw error;
  }
  return body;
}

function postJson(url, payload) {
  return requestJson(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

async function withRetry(request, attempts = 7) {
  for (let attempt = 1; ; attempt += 1) {
    try {
      return await request();
    } catch (error) {
      if (!error.retryable || attempt === attempts) throw error;
      setStatus(`The API is waking up, retrying in 10 seconds (${attempt}/${attempts - 1})…`);
      await new Promise((resolve) => setTimeout(resolve, 10000));
    }
  }
}

async function loadChart(imageId, url, payload) {
  const image = document.getElementById(imageId);
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    image.src = URL.createObjectURL(await response.blob());
    image.hidden = false;
  } catch (error) {
    const message = document.createElement('p');
    message.className = 'error';
    message.textContent = `Could not load this chart (${error.message}).`;
    image.replaceWith(message);
  }
}

function setStatus(message, isError = false) {
  const status = document.getElementById('status');
  status.textContent = message;
  status.hidden = !message;
  status.classList.toggle('error', isError);
}

function setText(id, value) {
  document.getElementById(id).textContent = value;
}

function renderResults(comments, predictions) {
  const counts = { '1': 0, '0': 0, '-1': 0 };
  predictions.forEach(({ sentiment }) => { counts[String(sentiment)] += 1; });
  const total = predictions.length;
  const averageScore = predictions.reduce((sum, { sentiment }) => sum + Number(sentiment), 0) / total;
  const totalWords = comments.reduce((sum, { text }) => sum + text.split(/\s+/).filter(Boolean).length, 0);

  setText('metric-total', total);
  setText('metric-unique', new Set(comments.map((c) => c.authorId)).size);
  setText('metric-length', `${(totalWords / total).toFixed(1)} words`);
  // Average of -1..1 rescaled to 0..10, as in the original extension
  setText('metric-score', `${(((averageScore + 1) / 2) * 10).toFixed(1)}/10`);

  document.querySelectorAll('.sentiment-box').forEach((box) => {
    const sentiment = box.dataset.sentiment;
    box.querySelector('.pct').textContent = `${((counts[sentiment] / total) * 100).toFixed(1)}%`;
    box.querySelector('.count').textContent = `${counts[sentiment]} comments`;
    box.addEventListener('click', () => showComments(predictions, sentiment));
  });
  document.getElementById('clear-filter').addEventListener('click', () => showComments(predictions, null));

  showComments(predictions, null);
  document.getElementById('results').hidden = false;
  return counts;
}

function showComments(predictions, sentiment) {
  const filtered = sentiment !== null;
  const items = filtered
    ? predictions.filter((p) => String(p.sentiment) === sentiment)
    : predictions.slice(0, TOP_N);

  setText('comments-title', filtered ? `${SENTIMENTS[sentiment].label} Comments (${items.length})` : `Top ${TOP_N} Comments`);
  document.getElementById('clear-filter').hidden = !filtered;
  document.querySelectorAll('.sentiment-box').forEach((box) => {
    box.classList.toggle('active', box.dataset.sentiment === sentiment);
  });
  document.getElementById('comment-list').replaceChildren(...items.map(commentItem));
  if (filtered) document.getElementById('comments-section').scrollIntoView({ behavior: 'smooth' });
}

function commentItem({ comment, sentiment }) {
  const { label, className } = SENTIMENTS[String(sentiment)];
  const item = document.createElement('li');
  item.className = `comment-item comment-item--${className}`;

  const tag = document.createElement('span');
  tag.className = `tag tag--${className}`;
  tag.textContent = label;

  const text = document.createElement('span');
  text.textContent = comment; // textContent, never innerHTML: comment text is untrusted input

  item.append(tag, text);
  return item;
}
