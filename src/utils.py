import os
import cv2
import glob
def images_to_video(image_folder: str, output_video_path: str, fps: int = 30):
    images = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))

    if not images:
        raise FileNotFoundError(f"У папці {image_folder} не знайдено жодного .jpg файлу!")

    first_frame = cv2.imread(images[0])
    height, width, _ = first_frame.shape

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Збираємо {len(images)} кадрів у відео...")
    for img_path in images:
        frame = cv2.imread(img_path)
        out.write(frame)

    out.release()
    print(f"Готово! Відео збережено: {output_video_path}")
