import requests

url = "https://res.cloudinary.com/dpfot18bn/image/upload/v1790990172/trylix/tryon/masks/hsjxvqjhtvpzkhvqkwjl.png"
r = requests.get(url, timeout=30)
print(f"Status: {r.status_code}")
print(f"Content-Type: {r.headers.get('content-type')}")
print(f"Length: {len(r.content)}")
print(f"First 100 bytes: {r.content[:100]}")

# Try without version number
url2 = "https://res.cloudinary.com/dpfot18bn/image/upload/trylix/tryon/masks/hsjxvqjhtvpzkhvqkwjl.png"
r2 = requests.get(url2, timeout=30)
print(f"\nWithout version - Status: {r2.status_code}")
print(f"Content-Type: {r2.headers.get('content-type')}")
print(f"Length: {len(r2.content)}")
