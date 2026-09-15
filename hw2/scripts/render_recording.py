"""MP4-воспроизведение check.cast с исходными временными метками, без монтажа."""
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFont


def main():
    root = Path(__file__).resolve().parents[1]
    records = [json.loads(line) for line in (root/'docs/check.cast').read_text().splitlines()]
    header, events = records[0], records[1:]
    if not shutil.which('ffmpeg'):
        raise SystemExit('Установите ffmpeg для экспорта видео.')
    fonts = ['/System/Library/Fonts/Menlo.ttc', '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf']
    font = next((ImageFont.truetype(p, 16) for p in fonts if Path(p).exists()), None)
    if font is None:
        raise SystemExit('Нужен моноширинный шрифт Menlo или DejaVuSansMono.')
    width, height = 1344, 944
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
               '-video_size', f'{width}x{height}', '-framerate', '1', '-i', '-', '-an',
               '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(root/'docs/check.mp4')]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    ansi = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
    text, cursor = '', 0
    try:
        for second in range(math.ceil(events[-1][0]) + 6):
            while cursor < len(events) and events[cursor][0] <= second:
                text += events[cursor][2]
                cursor += 1
            lines = ansi.sub('', text).replace('\r', '').split('\n')
            wrapped = [part for line in lines for part in
                       ([line[i:i+header['width']] for i in range(0,len(line),header['width'])] or [''])]
            frame = Image.new('RGB', (width,height), '#111827')
            draw = ImageDraw.Draw(frame)
            draw.text((28,14), f"Terminal replay | make check | {second}s", font=font, fill='#94a3b8')
            for row, line in enumerate(wrapped[-header['height']:]):
                color = '#86efac' if '✓' in line or 'Все проверки пройдены' in line else '#e5e7eb'
                draw.text((28, 48+row*21), line, font=font, fill=color)
            encoder.stdin.write(frame.tobytes())
    finally:
        encoder.stdin.close()
    if encoder.wait():
        raise SystemExit('ffmpeg завершился с ошибкой')
    print(root/'docs/check.mp4')


if __name__ == '__main__':
    main()
