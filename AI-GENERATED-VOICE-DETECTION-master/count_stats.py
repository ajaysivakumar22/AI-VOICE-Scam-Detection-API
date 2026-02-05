import os

ai_dir = r"d:\GuviHCL_Hackathon\ai_voice_detection_api\training\data\ai"
human_dir = r"d:\GuviHCL_Hackathon\ai_voice_detection_api\training\data\human"

ai_count = len([f for f in os.listdir(ai_dir) if os.path.isfile(os.path.join(ai_dir, f))])
human_count = 0
if os.path.exists(human_dir):
    human_count = len([f for f in os.listdir(human_dir) if os.path.isfile(os.path.join(human_dir, f))])

print(f"AI Samples: {ai_count}")
print(f"Human Samples: {human_count}")

if human_count > 0:
    ratio = ai_count / human_count
    print(f"Ratio (AI/Human): {ratio:.2f}")
    # Normalize to smallest integer ratio if possible, or just x:1
    print(f"Approximate Ratio: {ai_count}:{human_count}")
else:
    print("Ratio: Undefined (0 human samples)")
