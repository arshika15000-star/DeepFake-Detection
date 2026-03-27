# DeepFake Detector - Extension Icons

The extension needs icon files at three sizes. Since image generation is unavailable, 
the extension will still load without icons (you'll see the default puzzle piece icon in Chrome).

## How to add proper icons:
1. Create/paste any square image (e.g., from Canva or Paint)
2. Save as:
   - icons/icon16.png  (16x16px)
   - icons/icon48.png  (48x48px)
   - icons/icon128.png (128x128px)

Alternatively, update manifest.json to remove the icon references if you want to skip icons:
Replace "default_icon", "icons" sections with empty objects {}.
