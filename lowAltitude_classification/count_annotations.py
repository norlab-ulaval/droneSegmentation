from pathlib import Path
import re

pattern = r"^(.*?)\s*\d+\s*x\s*\d+"

annotated_path = Path("/data/annotated/Annotated_images")
annotated_files = [p.name for p in annotated_path.rglob("*x*.jpg")]
# print(annotated_files)
pic_names = [re.search(pattern, p).group(1) for p in annotated_files]
# print(pic_names)
print(len(set(pic_names)))
