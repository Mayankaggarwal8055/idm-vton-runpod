"""Submit RunPod job with chest-extended mask for the neckline fix test."""
import requests
import json
import time

RUNPOD_API_KEY = "rpa_PVJ5EBBWIIXMJT0UPES249IJDCX4WQ7BB4CEJ3V9h9vhvh"
ENDPOINT_ID = "lflfx4zpkxyy4d"

payload = {
    "input": {
        "person_image_url": "https://res.cloudinary.com/dpfot18bn/image/upload/v1790876545/trylix/tryon/gkkralk0kue5rjjgecn0.jpg",
        "garment_image_url": "https://res.cloudinary.com/dpfot18bn/image/upload/v1790883465/trylix/tryon/garments/xu8lh4iefq7fjdl8xjkj.jpg",
        "mask_image_url": "https://res.cloudinary.com/dpfot18bn/image/upload/v1790895429/trylix/tryon/masks/wark5qqcvoismxctilnx.png",
        "cloth_type": "upper_body",
        "num_inference_steps": 30,
        "guidance_scale": 2.0,
        "seed": 42,
        "mask_quality_score": 95.0,
        "garment_prompt": "navy blue button-down collared shirt with three-quarter sleeves"
    }
}

print("Submitting RunPod job...")
resp = requests.post(
    f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run",
    headers={
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    },
    json=payload,
    timeout=30
)
print(f"Status: {resp.status_code}")
result = resp.json()
print(json.dumps(result, indent=2))

job_id = result.get("id")
if not job_id:
    print("ERROR: No job ID returned")
    exit(1)

print(f"\nJob ID: {job_id}")
print("Polling for result...")

for i in range(60):
    time.sleep(5)
    status_resp = requests.get(
        f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{job_id}",
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
        timeout=30
    )
    status_data = status_resp.json()
    status = status_data.get("status", "UNKNOWN")
    print(f"  [{i*5}s] Status: {status}")
    
    if status == "COMPLETED":
        output = status_data.get("output", {})
        print("\n=== COMPLETED ===")
        print(json.dumps(output, indent=2)[:2000])
        
        # Extract result image URL
        result_url = None
        if isinstance(output, dict):
            result_url = output.get("result_url") or output.get("result_image") or output.get("image") or output.get("output_image")
            # Check nested
            if not result_url and "result" in output:
                r = output["result"]
                if isinstance(r, dict):
                    result_url = r.get("result_image") or r.get("image")
                elif isinstance(r, str) and r.startswith("http"):
                    result_url = r
        
        if result_url:
            print(f"\nResult URL: {result_url}")
            # Download result
            img_resp = requests.get(result_url, timeout=60)
            with open("tests/output_debug/chest_fixed_diffusion_raw.png", "wb") as f:
                f.write(img_resp.content)
            print("Saved: tests/output_debug/chest_fixed_diffusion_raw.png")
        else:
            print("WARNING: Could not find result image URL in output")
            # Save full output for inspection
            with open("tests/output_debug/chest_fixed_runpod_output.json", "w") as f:
                json.dump(status_data, f, indent=2)
            print("Saved full output to: tests/output_debug/chest_fixed_runpod_output.json")
        break
    
    elif status in ("FAILED", "CANCELLED", "TIMED_OUT"):
        print(f"\n=== {status} ===")
        print(json.dumps(status_data, indent=2)[:2000])
        break
else:
    print("\nTIMEOUT: Job did not complete in 300s")
