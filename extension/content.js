// Content script — runs on every page
// Checks if there's a pending URL from right-click context menu
chrome.runtime.sendMessage({ type: 'GET_PENDING_URL' }, (response) => {
  if (response && response.url) {
    // Store in sessionStorage so popup can pick it up
    sessionStorage.setItem('df_pending_url', response.url);
  }
});
