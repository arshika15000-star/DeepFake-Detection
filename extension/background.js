// Background service worker for DeepFake Detector extension

// Create context menu for right-clicking images
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'analyze-image',
    title: '🔍 Check for Deepfake',
    contexts: ['image'],
  });
  chrome.contextMenus.create({
    id: 'analyze-link',
    title: '🔍 Check this link for Deepfake',
    contexts: ['link'],
  });
});

// Handle right-click context menu
chrome.contextMenus.onClicked.addListener((info, tab) => {
  const url = info.srcUrl || info.linkUrl;
  if (!url) return;

  // Store URL in local storage and open popup
  chrome.storage.local.set({ pendingUrl: url }, () => {
    chrome.action.openPopup();
  });
});

// Listen for messages from popup or content script
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'GET_PENDING_URL') {
    chrome.storage.local.get(['pendingUrl'], (data) => {
      sendResponse({ url: data.pendingUrl || null });
      chrome.storage.local.remove('pendingUrl');
    });
    return true; // Keep channel open for async
  }
});
