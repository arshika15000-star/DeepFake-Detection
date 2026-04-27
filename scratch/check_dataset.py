import os

base = r"c:\Users\Vanshina Saxena\OneDrive\Desktop\Deepfake Detection\dataset\SDFVD\SDFVD"
real_path = os.path.join(base, "videos_real")
fake_path = os.path.join(base, "videos_fake")

real = os.listdir(real_path) if os.path.exists(real_path) else []
fake = os.listdir(fake_path) if os.path.exists(fake_path) else []

print(f"Real videos: {len(real)}")
print(f"Fake videos: {len(fake)}")
print(f"\nSample REAL filenames: {real[:5]}")
print(f"Sample FAKE filenames: {fake[:5]}")
