import cloudinary
import cloudinary.uploader

cloudinary.config(
    cloud_name="dpfot18bn",
    api_key="188815947465698",
    api_secret="GDIXa9G_uOe9Fv2DIUJIT6tapyc"
)

result = cloudinary.uploader.upload(
    "tests/output_debug/chest_extended_v2_mask.png",
    folder="trylix/tryon/masks",
    resource_type="image"
)
print("URL:", result["secure_url"])
print("Public ID:", result["public_id"])

# Verify
import requests
r = requests.get(result["secure_url"], timeout=30)
print(f"Verify: Status={r.status_code}, Length={len(r.content)}")
