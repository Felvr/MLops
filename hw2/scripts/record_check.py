"""Записать реальный make check в псевдотерминале (macOS/Linux)."""
import codecs
import errno
import fcntl
import json
import os
from pathlib import Path
import pty
import struct
import subprocess
import sys
import termios
import time


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    root.joinpath('docs').mkdir(exist_ok=True)
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 42, 132, 0, 0))
    env = dict(os.environ, TERM='xterm-256color', PYTHONUNBUFFERED='1')
    started = time.monotonic()
    decoder = codecs.getincrementaldecoder('utf-8')('replace')
    with open('docs/check.cast', 'w', encoding='utf-8') as cast, open('docs/check.txt', 'w', encoding='utf-8') as log:
        cast.write(json.dumps({'version': 2, 'width': 132, 'height': 42,
                               'timestamp': int(time.time()), 'command': 'make check',
                               'title': 'ДЗ2: реальный make check'}, ensure_ascii=False) + '\n')

        def emit(text):
            cast.write(json.dumps([round(time.monotonic()-started, 4), 'o', text], ensure_ascii=False)+'\n')
            cast.flush()
            log.write(text.replace('\r', ''))
            log.flush()
            print(text, end='', flush=True)

        emit('$ make check\r\n')
        proc = subprocess.Popen(['make', 'check'], stdin=slave, stdout=slave, stderr=slave, env=env)
        os.close(slave)
        try:
            while True:
                try:
                    chunk = os.read(master, 65536)
                except OSError as error:
                    if error.errno == errno.EIO:
                        break
                    raise
                if not chunk:
                    break
                emit(decoder.decode(chunk))
        finally:
            os.close(master)
        tail = decoder.decode(b'', final=True)
        if tail:
            emit(tail)
        code = proc.wait()
        emit(f'\r\nProcess exit code: {code}\r\n')
    return code


if __name__ == '__main__':
    sys.exit(main())
